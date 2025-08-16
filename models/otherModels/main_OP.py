"""
benchmark_models.py  ·  Benchmark EEG (orchestrator-ready, schema-compatible)

- Emite UN [RESULT_JSON] POR MODELO con el MISMO schema que train.py (result_summary).
  Claves no aplicables se devuelven como null (None).
- Artefactos por modelo en --exp-dir/<run_id>/<model>/: best.pt, last.pt, history.csv, metrics.json, args.json
- CSV global acumulado (--out_csv) con run_id × model × split
- Progreso por época: [EPOCH_JSON] {...}
- Resumen global del run: [RUN_SUMMARY_JSON] {...}

Ejemplo:
python benchmark_models.py \
  --dataset_file data/processed/BCI2020/filtered_BCI2020.npz \
  --dataset BCI --device cuda:0 --epochs 50 \
  --models eegnet shallowconvnet \
  --exp-dir runs/bench --out_csv runs/bench/results.csv \
  --results_jsonl runs/bench/all.jsonl
"""
from __future__ import annotations

import argparse, csv, hashlib, json, os, random, signal, sys, traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Callable, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from torch.utils.data import DataLoader

# --- Repos propios ---
from utils import load_data_BCI, load_data_TOL, get_dataloader


# ========================= Helpers orquestador / reproducibilidad =========================

def _as_scalar(x):
    if isinstance(x, (float, int)): return x
    if isinstance(x, (np.floating, np.integer)): return float(x)
    return x

def safe_jsonify(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = safe_jsonify(v)
        elif hasattr(v, "tolist"):
            out[k] = v.tolist()
        else:
            out[k] = _as_scalar(v)
    return out

def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]

def array_fingerprint(a: np.ndarray, sample_bytes: int = 0) -> str:
    if sample_bytes and a.nbytes > sample_bytes:
        buf = memoryview(a).tobytes()[:sample_bytes]
    else:
        buf = memoryview(a).tobytes()
    return hashlib.sha1(buf).hexdigest()[:10]

# ==================================== Model Zoo ====================================

import inspect

def _save_args(init_fn: Callable):
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


# ================================== Dataset & utils ==================================

def build_loaders(
    dataset_file: Path,
    subject_id: int | None,
    batch_train: int,
    batch_eval: int,
    dataset: str,
    shuffle: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:

    print(f"[DATASET] Loading dataset from: {dataset_file.name}")

    if dataset == "BCI":
        print("[data] Cargando datos del dataset BCI")
        Xtr, ytr, Xva, yva, Xte, yte = load_data_BCI(dataset_file=dataset_file, subject_id=subject_id)
    elif dataset == "TOL":
        print("[data] Cargando datos del dataset TOL")
        Xtr, ytr, Xva, yva, Xte, yte = load_data_TOL(dataset_file=dataset_file, subject_id=subject_id)
    else:
        raise ValueError(f"Dataset '{dataset}' no soportado. Usa 'BCI' o 'TOL'.")

    meta = {
        "dataset_file": str(dataset_file.resolve()),
        "subject_id": subject_id,
        "n_samples": {"train": int(len(ytr)), "val": int(len(yva)), "test": int(len(yte))},
        "fingerprints": {
            "X_train": array_fingerprint(Xtr, sample_bytes=2_000_000),
            "y_train": array_fingerprint(ytr),
            "X_val":   array_fingerprint(Xva, sample_bytes=2_000_000),
            "y_val":   array_fingerprint(yva),
            "X_test":  array_fingerprint(Xte, sample_bytes=2_000_000),
            "y_test":  array_fingerprint(yte),
        },
    }

    tr, va, te = get_dataloader(Xtr, ytr, Xva, yva, Xte, yte, batch_train, batch_eval, shuffle=shuffle)
    return tr, va, te, meta


# ============================== Train / Eval genéricos ==============================


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, device, scaler=None):
    model.train()
    running_loss, preds, targets = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item()) * x.size(0)
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
            running_loss += float(loss.item()) * x.size(0)
            preds.append(logits.cpu())
            targets.append(y.cpu())
    return _gather_metrics(running_loss, preds, targets, len(loader.dataset))

def _gather_metrics(total_loss, preds_tensors, target_tensors, n_samples):
    logits = torch.cat(preds_tensors)
    preds = logits.softmax(dim=1).argmax(dim=1).numpy()
    targets = torch.cat(target_tensors).numpy()
    return {
        "loss": total_loss / max(1, n_samples),
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro")),
        "macro_recall": float(recall_score(targets, preds, average="macro")),
        "macro_precision": float(precision_score(targets, preds, average="macro", zero_division=0)),
        "y_true": targets,        # para cm en test
        "y_pred": preds,
    }


# ========================================= Run =========================================

def run_benchmark(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    # Datos
    train_loader, val_loader, test_loader, data_meta = build_loaders(
        dataset_file=Path(args.dataset_file),
        subject_id=args.subject_id,
        batch_train=args.batch_train,
        batch_eval=args.batch_eval,
        dataset=args.dataset,
    )

    # CSV global
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        with out_path.open("w", newline="", encoding="utf-8") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["run_id","model","split","accuracy","macro_f1","macro_recall","macro_precision","loss"])

    interrupted = {"flag": False}
    def _handle_sigint(signum, frame):
        interrupted["flag"] = True
        print("[signal] SIGINT recibido: saliendo tras la época actual…", file=sys.stderr)
    signal.signal(signal.SIGINT, _handle_sigint)

    # Run id & dir
    now = datetime.now()
    run_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{short_hash(str(vars(args)))}"
    base_exp_dir = Path(args.exp_dir) / run_id
    base_exp_dir.mkdir(parents=True, exist_ok=True)

    def _append_csv(model_name: str, split: str, m: dict):
        with out_path.open("a", newline="", encoding="utf-8") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([run_id, model_name, split,
                             f"{m['accuracy']:.4f}", f"{m['macro_f1']:.4f}",
                             f"{m['macro_recall']:.4f}", f"{m['macro_precision']:.4f}",
                             f"{m['loss']:.5f}"])

    run_started = datetime.now().timestamp()
    models_summaries = []

    for name in args.models:
        assert name in MODEL_ZOO, f"Modelo '{name}' no reconocido. Opciones: {list(MODEL_ZOO)}"
        print(f"\n▶ Entrenando {name} …")

        model = MODEL_ZOO[name](args.n_channels, args.n_time, args.n_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()

        # Artefactos
        model_dir = base_exp_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt = model_dir / "best.pt"
        last_ckpt = model_dir / "last.pt"
        write_json(model_dir / "args.json", {
            "run_id": run_id, "model": name,
            "training": {"epochs": args.epochs, "lr": args.lr, "batch_train": args.batch_train,
                         "batch_eval": args.batch_eval, "seed": args.seed},

            })

        # Historia
        history_rows = []
        best_val_acc = -1.0
        best_epoch = None
        tr_last = None; va_last = None

        for epoch in range(1, args.epochs + 1):
            tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
            va = evaluate(model, val_loader, criterion, device)
            tr_last, va_last = tr, va

            # last ckpt
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "val_metrics": va}, last_ckpt)

            # best ckpt
            if va["accuracy"] > best_val_acc:
                best_val_acc = va["accuracy"]
                best_epoch = epoch
                torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(), "val_metrics": va}, best_ckpt)

            row = {
                "epoch": epoch,
                "train_acc": tr["accuracy"], "train_macro_f1": tr["macro_f1"],
                "train_macro_recall": tr["macro_recall"], "train_macro_precision": tr["macro_precision"],
                "train_loss": tr["loss"],
                "val_acc": va["accuracy"], "val_macro_f1": va["macro_f1"],
                "val_macro_recall": va["macro_recall"], "val_macro_precision": va["macro_precision"],
                "val_loss": va["loss"],
                "lr": float(optimizer.param_groups[0]["lr"]),
                "interrupted": bool(interrupted["flag"]),
            }
            history_rows.append(row)
            print("[EPOCH_JSON]", json.dumps(safe_jsonify({"run_id": run_id, "model": name, **row}), ensure_ascii=False))

            if interrupted["flag"]:
                print("[train] Interrupción recibida. Rompiendo bucle de épocas…")
                break

        # Restaurar best y evaluar test
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(ckpt["state_dict"])
        te = evaluate(model, test_loader, criterion, device)
        cm = confusion_matrix(te["y_true"], te["y_pred"]).tolist() if "y_true" in te else None

        # Guardar history.csv
        if history_rows:
            hist_path = model_dir / "history.csv"
            with hist_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
                writer.writeheader(); writer.writerows(history_rows)

        # Guardar metrics.json (modelo)
        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = total_params * 4 / (1024 ** 2)  # float32
        metrics_payload = {
            "model": name,
            "params_total": int(total_params),
            "model_size_mb": float(model_size_mb),
            "best_epoch": int(best_epoch) if best_epoch else None,
            "val_best_accuracy": float(best_val_acc if best_val_acc >= 0 else 0.0),
            "train_last": {k: _as_scalar(v) for k,v in tr_last.items()} if tr_last else None,
            "val_last":   {k: _as_scalar(v) for k,v in va_last.items()} if va_last else None,
            "test":       {k: _as_scalar(v) for k,v in te.items()},
            "paths": {"best_ckpt": str(best_ckpt.resolve()),
                      "last_ckpt": str(last_ckpt.resolve()),
                      "history_csv": str((model_dir / "history.csv").resolve())},
        }
        write_json(model_dir / "metrics.json", safe_jsonify(metrics_payload))

        # CSV global
        _append_csv(name, "train", tr_last)
        _append_csv(name, "val",   va_last)
        _append_csv(name, "test",  te)

        # ======================== EMISIÓN COMPATIBLE CON train.py ========================
        # Construimos result_summary con EXACTAS claves (valores n/a = None)
        duration_sec = datetime.now().timestamp() - run_started
        result_summary = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_id": run_id,
            "exp_dir": str(base_exp_dir.resolve()),
            "dataset": args.dataset,
            "dataset_file": data_meta["dataset_file"],
            "subject_id": args.subject_id,
            "seed": args.seed,
            "device": args.device,
            # Modelo & HPs (algunos no aplican en benchmark → None)
            "num_epochs": args.epochs,
            "batch_train": args.batch_train,
            "batch_eval": args.batch_eval,
            "alpha": None,
            "num_classes": args.n_classes,
            "channels": args.n_channels,
            "n_T": None,
            "ddpm_dim": None,
            "encoder_dim": None,
            "fc_dim": None,
            # Info modelo
            "params_total": int(total_params),
            "model_size_mb": float(model_size_mb),
            "flops": None,
            # Validación (mejor)
            "best_epoch": int(best_epoch) if best_epoch else None,
            "val_best_accuracy": float(best_val_acc if best_val_acc >= 0 else 0.0),
            "val_best_macro_f1": float(va_last["macro_f1"]) if va_last else 0.0,
            # Test (del mejor checkpoint)
            "test_accuracy": float(te["accuracy"]),
            "test_macro_f1": float(te["macro_f1"]),
            "test_precision_macro": float(te["macro_precision"]),
            "test_recall_macro": float(te["macro_recall"]),
            # Matriz de confusión
            "confusion_matrix": cm,
            # Misc
            "duration_sec": float(duration_sec),
            "status": "interrupted" if interrupted["flag"] else "ok",
            # Extra no usado por tu tabla pero útil:
            "model_name": name,
        }
        # Emitimos UNA LÍNEA por modelo con la etiqueta esperada:
        print("[RESULT_JSON]", json.dumps(safe_jsonify(result_summary), ensure_ascii=False))
        if args.results_jsonl:
            append_jsonl(Path(args.results_jsonl), safe_jsonify(result_summary))

        models_summaries.append(result_summary)

        print(f"✓ {name} terminado. Test acc = {te['accuracy']:.3f}")
        if interrupted["flag"]:
            break

    # Resumen global del run (opcional)
    run_summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "exp_dir": str(base_exp_dir.resolve()),
        "dataset": args.dataset,
        "dataset_file": data_meta["dataset_file"],
        "subject_id": args.subject_id,
        "seed": args.seed,
        "device": args.device,
        "status": "interrupted" if interrupted["flag"] else "ok",
        "models": [m.get("model_name") for m in models_summaries],
    }
    print("[RUN_SUMMARY_JSON]", json.dumps(safe_jsonify(run_summary), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Benchmark de modelos para EEG (schema-compatible)")
    data = parser.add_argument_group("dataset")
    data.add_argument("--dataset_file", type=str, required=True, help="Path a .npz con splits")
    data.add_argument("--dataset", type=str, required=True, help="BCI | TOL")
    data.add_argument("--subject_id", type=int, default=None)
    data.add_argument("--n_channels", type=int, default=64)
    data.add_argument("--n_time", type=int, default=800)
    data.add_argument("--n_classes", type=int, default=5)

    train_opts = parser.add_argument_group("training")
    train_opts.add_argument("--epochs", type=int, default=150)
    train_opts.add_argument("--lr", type=float, default=1e-3)
    train_opts.add_argument("--weight_decay", type=float, default=0.0)
    train_opts.add_argument("--batch_train", type=int, default=64)
    train_opts.add_argument("--batch_eval", type=int, default=32)

    misc = parser.add_argument_group("misc")
    misc.add_argument("--models", nargs="*", default=list(MODEL_ZOO.keys()))
    misc.add_argument("--device", type=str, default="cpu")
    misc.add_argument("--seed", type=int, default=42)
    misc.add_argument("--out_csv", type=str, default="benchmark_results.csv")
    misc.add_argument("--exp-dir", type=str, default="runs/benchmarks")
    misc.add_argument("--results_jsonl", type=str, default=None)

    args = parser.parse_args()

    try:
        run_benchmark(args)
        sys.exit(0)
    except KeyboardInterrupt:
        err = {"timestamp": datetime.now().isoformat(timespec="seconds"),
               "status": "interrupted", "error_message": "KeyboardInterrupt"}
        print("[RUN_SUMMARY_JSON]", json.dumps(safe_jsonify(err), ensure_ascii=False))
        sys.exit(130)
    except Exception as e:
        err = {"timestamp": datetime.now().isoformat(timespec="seconds"),
               "status": "error", "error_message": f"{type(e).__name__}: {e}",
               "traceback": traceback.format_exc(limit=2)}
        print("[RUN_SUMMARY_JSON]", json.dumps(safe_jsonify(err), ensure_ascii=False))
        sys.exit(1)

if __name__ == "__main__":
    main()
