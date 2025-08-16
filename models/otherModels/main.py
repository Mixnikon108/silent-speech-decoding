"""
benchmark_models.py  ·  Pipeline de benchmarking para modelos de clasificación EEG
-------------------------------------------------------------------------------
Este script entrena y evalúa un listado configurable de modelos – desde baselines
simples hasta arquitecturas profundas comunes en EEG – sobre un dataset con las
siguientes características por defecto:
  * Frecuencia de muestreo: 256 Hz
  * Canales:              64
  * Ventana / trial:      800 muestras (≈ 3.1 s)
  * Nº de clases:         4

Entrada esperada
---------------
Se admite cualquiera de las dos opciones:
1) Un fichero *.npz* con los splits «X_train, y_train, X_val, y_val, X_test,
   y_test» (tal y como los produce `utils.load_data` del repo Diff‑E)
2) Path a las *mat files* + especificación de «subject» y «session» para usar la
   función `utils.load_data(root_dir, subject, session)` propia de Diff‑E.

Ejemplo de uso (split ya preparado):
    python benchmark_models.py \ 
        --dataset_file data/processed/my_dataset.npz \ 
        --device cuda:0 --epochs 200

Ejemplo de uso (archivos *.mat*):
    python benchmark_models.py \ 
        --root_dir data/raw/BCI2020/ --subject 2 --session 1 \
        --device cuda:0 --epochs 200

El script imprime y guarda (CSV) las métricas *accuracy*, *macro‑F1*, *macro‑
recall* y *cross‑entropy loss* para *train*, *val* y *test* de cada modelo.  

Autor: ChatGPT o3
"""
from __future__ import annotations

import argparse
import csv
import inspect
import random
from pathlib import Path
from typing import Dict, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader
from utils import load_data_BCI, load_data_TOL, get_dataloader


# -----------------------------------------------------------------------------
# ▼ Model Zoo – implementaciones ligeras y autosuficientes
# -----------------------------------------------------------------------------

def _save_args(init_fn: Callable):
    """Decorator que almacena kwargs como atributos (para reproducibilidad)."""
    def wrapper(self, *args, **kwargs):
        for name, val in inspect.signature(init_fn).bind(self, *args, **kwargs).arguments.items():
            if name != "self":
                setattr(self, name, val)
        init_fn(self, *args, **kwargs)
    return wrapper

class MLPClassifier(nn.Module):
    @_save_args
    def __init__(self, n_channels: int = 64, n_time: int = 800, n_classes: int = 5):
        super().__init__()
        in_dim = n_channels * n_time
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.net(x)

class SimpleCNN(nn.Module):
    @_save_args
    def __init__(self, n_channels: int = 64, n_time: int = 800, n_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=8, padding=4),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=8, padding=4),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=8, padding=4),
            nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.squeeze(-1))

class EEGNet(nn.Module):
    """Versión PyTorch del EEGNet compacta (Lawhern et al., 2018)."""
    @_save_args
    def __init__(self, n_channels: int = 64, n_time: int = 800, n_classes: int = 5, F1: int = 8, D: int = 2, F2_scale: int = 2, dropout: float = 0.25):
        super().__init__()
        F2 = F1 * D * F2_scale
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 16), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )
        # cálculo dinámico del tamaño plano
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_time)
            out = self.separable(self.depthwise(self.firstconv(dummy)))
            self.flat_dim = out.numel()
        self.classifier = nn.Linear(self.flat_dim, n_classes)

    def forward(self, x):
        # Expect: [B, C, T] → [B, 1, C, T]
        x = x.unsqueeze(1)
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)



class Square(nn.Module):
    def forward(self, x):
        return torch.pow(x, 2)

class Log(nn.Module):
    def forward(self, x):
        # Añade una pequeña constante para evitar log(0)
        return torch.log(torch.clamp(x, min=1e-6))

class ShallowConvNet(nn.Module):
    @_save_args
    def __init__(self, n_channels: int = 64, n_time: int = 800, n_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), bias=False),
            nn.Conv2d(40, 40, (n_channels, 1), groups=40, bias=False),
            nn.BatchNorm2d(40),
            Square(),  # custom op defined below
            nn.AvgPool2d((1, 75), stride=(1, 15)),
            Log(),    # custom op
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_time)
            out = self.features(dummy)
            self.flat_dim = out.numel()
        self.classifier = nn.Linear(self.flat_dim, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class DeepConvNet(nn.Module):
    @_save_args
    def __init__(self, n_channels: int = 64, n_time: int = 800, n_classes: int = 5):
        super().__init__()
        def block(in_c, out_c, kernel, pool=2):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, (1, kernel), bias=False),
                nn.BatchNorm2d(out_c),
                nn.ELU(),
                nn.MaxPool2d((1, pool)),
                nn.Dropout(0.25),
            )
        self.net = nn.Sequential(
            nn.Conv2d(1, 25, (1, 5), bias=False),
            nn.Conv2d(25, 25, (n_channels, 1), groups=25, bias=False),
            nn.BatchNorm2d(25), nn.ELU(), nn.MaxPool2d((1, 2)), nn.Dropout(0.25),
            block(25, 50, 5), block(50, 100, 5), block(100, 200, 5),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_time)
            out = self.net(dummy)
            self.flat_dim = out.numel()
        self.classifier = nn.Linear(self.flat_dim, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

MODEL_ZOO: Dict[str, Callable[..., nn.Module]] = {
    "mlp": MLPClassifier,
    "cnn": SimpleCNN,
    "eegnet": EEGNet,
    "shallowconvnet": ShallowConvNet,
    "deepconvnet": DeepConvNet,
}

# -----------------------------------------------------------------------------
# ▼ Datasets & utils
# -----------------------------------------------------------------------------

def build_loaders(
    dataset_file: Path,
    subject_id: int | None,
    batch_train: int,
    batch_eval: int,
    dataset: str,
    shuffle: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    print(f"[DATASET] Loading dataset from: {dataset_file.name}")  # Only the file name

    if dataset == "BCI":
        print("[data] Cargando datos del dataset BCI")
        X_train, y_train, X_val, y_val, X_test, y_test = load_data_BCI(dataset_file=dataset_file, subject_id=subject_id)
    elif dataset == "TOL":
        print("[data] Cargando datos del dataset TOL")
        X_train, y_train, X_val, y_val, X_test, y_test = load_data_TOL(dataset_file=dataset_file, subject_id=subject_id)


    return get_dataloader(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        batch_train,
        batch_eval,
        shuffle=shuffle,
    )

# -----------------------------------------------------------------------------
# ▼ Entrenamiento y evaluación genéricos
# -----------------------------------------------------------------------------

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, device):
    model.train()
    running_loss, preds, targets = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        preds.append(logits.detach().cpu())
        targets.append(y.detach().cpu())
    return _gather_metrics(running_loss, preds, targets, len(loader.dataset))

def evaluate(model: nn.Module, loader: DataLoader, criterion, device):
    model.eval()
    running_loss, preds, targets = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item() * x.size(0)
            preds.append(logits.cpu())
            targets.append(y.cpu())
    return _gather_metrics(running_loss, preds, targets, len(loader.dataset))

def _gather_metrics(total_loss, preds_tensors, target_tensors, n_samples):
    preds = torch.cat(preds_tensors).softmax(dim=1).argmax(dim=1).numpy()
    targets = torch.cat(target_tensors).numpy()
    metrics = {
        "loss": total_loss / n_samples,
        "accuracy": accuracy_score(targets, preds),
        "macro_f1": f1_score(targets, preds, average="macro"),
        "macro_recall": recall_score(targets, preds, average="macro"),
    }
    return metrics

# -----------------------------------------------------------------------------
# ▼ Main
# -----------------------------------------------------------------------------

def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Load data
    train_loader, val_loader, test_loader = build_loaders(
        dataset_file=Path(args.dataset_file),
        subject_id=args.subject_id,
        batch_train=args.batch_train,
        batch_eval=args.batch_eval,
        dataset=args.dataset,
    )

    # Infieres dimensiones automáticamente:
    n_channels = args.n_channels        # número de canales
    n_time     = args.n_time        # longitud temporal (muestras por trial)
    n_classes  = args.n_classes  # número de clases (si los labels son 0-indexed)



    # Prepare CSV output
    out_path = Path(args.out_csv)
    if out_path.exists():
        out_path.unlink()
    with out_path.open("w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["model", "split", "accuracy", "macro_f1", "macro_recall", "loss"])

    # Loop over models
    for name in args.models:
        assert name in MODEL_ZOO, f"Modelo '{name}' no reconocido. Opciones: {list(MODEL_ZOO)}"
        print(f"\n▶ Entrenando {name} …")
        model = MODEL_ZOO[name](n_channels, n_time, n_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        for epoch in range(1, args.epochs + 1):
            tr_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate(model, val_loader, criterion, device)
            if val_metrics["accuracy"] > best_val_acc:
                best_model_state = model.state_dict()
                best_val_acc = val_metrics["accuracy"]
            if epoch % args.print_every == 0 or epoch == args.epochs:
                print(f"Epoch {epoch:03}/{args.epochs} – "
                      f"train acc: {tr_metrics['accuracy']:.3f}, val acc: {val_metrics['accuracy']:.3f}")
        # Restore best model
        model.load_state_dict(best_model_state)
        test_metrics = evaluate(model, test_loader, criterion, device)

        # Save metrics
        def _write_row(split, m):
            with out_path.open("a", newline="") as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow([name, split, f"{m['accuracy']:.4f}", f"{m['macro_f1']:.4f}", f"{m['macro_recall']:.4f}", f"{m['loss']:.5f}"])
        _write_row("train", tr_metrics)
        _write_row("val", val_metrics)
        _write_row("test", test_metrics)
        print(f"✓ {name} terminado. Test acc = {test_metrics['accuracy']:.3f}\n")

    print(f"\nBenchmark completo → resultados en {out_path}")

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark de modelos para EEG")
    data = parser.add_argument_group("dataset")
    data.add_argument("--dataset_file", type=str, default=None, help="Path a .npz con splits pre‑hechos")
    parser.add_argument("--dataset", type=str, required=True)
    data.add_argument("--subject_id", type=int, default=None)
    data.add_argument("--n_channels", type=int, default=64)
    data.add_argument("--n_time", type=int, default=800)
    data.add_argument("--n_classes", type=int, default=5)


    train_opts = parser.add_argument_group("training")
    train_opts.add_argument("--epochs", type=int, default=150)
    train_opts.add_argument("--batch_size", type=int, default=32)
    train_opts.add_argument("--lr", type=float, default=1e-3)
    train_opts.add_argument("--print_every", type=int, default=10)
    train_opts.add_argument("--batch_train", type=int, default=64)
    train_opts.add_argument("--batch_eval", type=int, default=32)

    misc = parser.add_argument_group("misc")
    misc.add_argument("--models", nargs="*", default=list(MODEL_ZOO.keys()), help="Lista de modelos a evaluar")
    misc.add_argument("--device", type=str, default="cpu")
    misc.add_argument("--seed", type=int, default=42)
    misc.add_argument("--out_csv", type=str, default="benchmark_results.csv")

    main(parser.parse_args())
