from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader
from utils import load_data_BCI, load_data_TOL, get_dataloader

# ------------------------------
# Modelo: SimpleCNN 1D (C x T)
# ------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, n_channels: int = 64, n_time: int = 800, n_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=8, padding=4, bias=True),  # W: (32, C, 8)
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=8, padding=4, bias=True),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=8, padding=4, bias=True),
            nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):               # x: [B, C, T]
        x = self.features(x)            # -> [B, 128, 1]
        return self.classifier(x.squeeze(-1))  # -> [B, n_classes]

# ------------------------------
# Data loaders
# ------------------------------
def build_loaders(dataset: str, dataset_file: Path, subject_id: int | None,
                  batch_train: int, batch_eval: int, shuffle: bool = True):
    if dataset == "BCI":
        Xtr, ytr, Xva, yva, Xte, yte = load_data_BCI(dataset_file=dataset_file, subject_id=subject_id)
    elif dataset == "TOL":
        Xtr, ytr, Xva, yva, Xte, yte = load_data_TOL(dataset_file=dataset_file, subject_id=subject_id)
    else:
        raise ValueError("dataset debe ser 'BCI' o 'TOL'")
    return get_dataloader(Xtr, ytr, Xva, yva, Xte, yte, batch_train, batch_eval, shuffle=shuffle)

# ------------------------------
# Entrenamiento / evaluación
# ------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device):
    model.eval()
    probs_list, y_list, loss_sum, n = [], [], 0.0, 0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        probs_list.append(logits.softmax(dim=1).cpu())
        y_list.append(y.cpu())
        n += x.size(0)
    probs = torch.cat(probs_list)
    y_true = torch.cat(y_list).numpy()
    y_pred = probs.argmax(dim=1).numpy()
    metrics = {
        "loss": loss_sum / n,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "macro_recall": recall_score(y_true, y_pred, average="macro"),
    }
    return metrics

def train(model, train_loader, val_loader, device, epochs=150, lr=1e-3, print_every=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_state, best_val_acc = None, -1.0
    for ep in range(1, epochs+1):
        model.train()
        loss_sum, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            n += x.size(0)
        if ep % print_every == 0 or ep == epochs:
            tr = {"loss": loss_sum / n}  # métricas rápidas
            va = evaluate(model, val_loader, device)
            if va["accuracy"] > best_val_acc:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_val_acc = va["accuracy"]
            print(f"Epoch {ep:03d}/{epochs}  |  train loss {tr['loss']:.4f}  |  val acc {va['accuracy']:.3f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# ------------------------------
# Explicabilidad por canal
# ------------------------------
def firstconv_channel_weight_scores(model: SimpleCNN):
    """
    Devuelve importancia por canal basada en la 1ª conv:
      - l1: suma |W| sobre filtros y kernel temporal
      - l2: norma L2 sobre filtros y kernel temporal
    """
    conv1: nn.Conv1d = None
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            conv1 = m
            break
    if conv1 is None:
        raise RuntimeError("No se encontró la primera Conv1d")
    W = conv1.weight.detach().cpu().numpy()  # [out=32, C, K]
    l1 = np.sum(np.abs(W), axis=(0, 2))      # -> [C]
    l2 = np.sqrt(np.sum(W**2, axis=(0, 2)))  # -> [C]
    return l1, l2

def _forward_batches(model, loader, device):
    """ Devuelve logits y x en CPU (sin grad). """
    model.eval()
    X_list, y_list, logits_list = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            logits_list.append(logits.cpu())
            X_list.append(x.cpu())
            y_list.append(y)  # CPU ya
    X = torch.cat(X_list)            # [N, C, T]
    y = torch.cat(y_list)            # [N]
    logits = torch.cat(logits_list)  # [N, K]
    return X, y, logits

def grad_saliency_per_channel(model, loader, device, method="saliency", max_batches=None):
    """
    method in {"saliency","gradxinput"}
    Devuelve importancia agregada por canal (y opcionalmente por tiempo) sobre el split de 'loader'.
    - saliency: mean_{samples,time} |d logit_y / dx|
    - gradxinput: mean |x * d logit_y / dx|
    """
    model.eval()
    total_ch = None
    total_ch_time = None
    n_seen = 0

    for b_idx, (x, y) in enumerate(loader):
        if (max_batches is not None) and (b_idx >= max_batches):
            break
        x = x.to(device).requires_grad_(True)      # [B, C, T]
        y = y.to(device)
        logits = model(x)                          # [B, K]
        # tomar logit de la clase verdadera (si quisieras target específico, cámbialo)
        selected = logits.gather(1, y.view(-1,1)).squeeze(1)
        grads = torch.autograd.grad(selected.sum(), x, retain_graph=False, create_graph=False)[0]  # [B, C, T]

        if method == "saliency":
            contrib = grads.abs()                  # |∂logit/∂x|
        elif method == "gradxinput":
            contrib = (grads * x).abs()            # |x * ∂logit/∂x|
        else:
            raise ValueError("method debe ser 'saliency' o 'gradxinput'")

        # Agregados
        ch_imp = contrib.mean(dim=(0, 2)).detach().cpu().numpy()     # [C]
        ch_time = contrib.mean(dim=0).detach().cpu().numpy()         # [C, T]

        if total_ch is None:
            total_ch = ch_imp
            total_ch_time = ch_time
        else:
            total_ch += ch_imp
            total_ch_time += ch_time

        n_seen += 1

    total_ch /= max(1, n_seen)
    total_ch_time /= max(1, n_seen)
    return total_ch, total_ch_time  # ([C], [C,T])

@torch.no_grad()
def permutation_importance_channels(model, val_loader, device, repeats=1, seed=42):
    """
    Permuta un canal a la vez (entre trials del batch) y mide la caída de accuracy en val.
    No re-entrena. Devuelve DataFrame-like dict con acc_base, acc_perm, delta_acc.
    """
    rng = np.random.default_rng(seed)
    # Métrica base
    base = evaluate(model, val_loader, device)["accuracy"]

    # Para evitar una pasada por canal por cada batch, colectamos primero todo val.
    Xv_list, yv_list = [], []
    for x, y in val_loader:
        Xv_list.append(x)
        yv_list.append(y)
    Xv = torch.cat(Xv_list)  # [N,C,T]
    yv = torch.cat(yv_list)

    N, C, T = Xv.shape
    results = []
    for ch in range(C):
        accs = []
        for r in range(repeats):
            idx = rng.permutation(N)
            Xp = Xv.clone()
            Xp[:, ch, :] = Xv[idx, ch, :]  # permuta canal ch entre trials (rompe su identidad)
            # eval rápida en batches
            bs = 256
            preds, targets = [], []
            for i in range(0, N, bs):
                xb = Xp[i:i+bs].to(device)
                yb = yv[i:i+bs].to(device)
                logits = model(xb)
                preds.append(logits.softmax(dim=1).argmax(dim=1).cpu())
                targets.append(yb.cpu())
            y_pred = torch.cat(preds).numpy()
            y_true = torch.cat(targets).numpy()
            accs.append(accuracy_score(y_true, y_pred))
        acc_perm = float(np.mean(accs))
        results.append({"channel": ch, "acc_base": base, "acc_perm": acc_perm, "delta_acc": base - acc_perm})
    return results

# ------------------------------
# Utilidades
# ------------------------------
def write_csv(path: Path, rows: list[dict]):
    if not rows: return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    parser = argparse.ArgumentParser(description="Entrenar CNN + explicabilidad por canal")
    parser.add_argument("--dataset", type=str, required=True, choices=["BCI","TOL"])
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--subject_id", type=int, default=None)
    parser.add_argument("--n_channels", type=int, default=64)
    parser.add_argument("--n_time", type=int, default=800)
    parser.add_argument("--n_classes", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_train", type=int, default=64)
    parser.add_argument("--batch_eval", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--outdir", type=str, default="./cnn_explain")
    parser.add_argument("--saliency_batches", type=int, default=None, help="máx. batches para saliency (None=todos)")
    parser.add_argument("--perm_repeats", type=int, default=1)
    args = parser.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Datos
    train_loader, val_loader, test_loader = build_loaders(
        dataset=args.dataset,
        dataset_file=Path(args.dataset_file),
        subject_id=args.subject_id,
        batch_train=args.batch_train,
        batch_eval=args.batch_eval,
        shuffle=True,
    )

    # Modelo
    model = SimpleCNN(args.n_channels, args.n_time, args.n_classes).to(device)

    # Entrenar
    model = train(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr, print_every=10)

    # Métricas finales
    tr = evaluate(model, train_loader, device)
    va = evaluate(model, val_loader, device)
    te = evaluate(model, test_loader, device)
    rows = []
    for split, m in [("train", tr), ("val", va), ("test", te)]:
        rows.append({"split": split, **{k: f"{v:.6f}" for k, v in m.items()}})
    write_csv(outdir / "metrics.csv", rows)
    print(f"[METRICS] train/val/test guardadas en {outdir/'metrics.csv'}")

    # ---------- Explicabilidad ----------
    # (1) Pesos de la primera conv → importancia por canal
    l1, l2 = firstconv_channel_weight_scores(model)
    w_rows = [{"channel": i, "w_l1": float(l1[i]), "w_l2": float(l2[i])} for i in range(len(l1))]
    write_csv(outdir / "weights_firstconv_channel_importance.csv", w_rows)
    print("[EXPLAIN] weights_firstconv_channel_importance.csv")

    # (2) Saliency |∂logit/∂x| y (3) Grad×Input |x·∂logit/∂x|
    sal_ch, sal_ch_time = grad_saliency_per_channel(model, val_loader, device, method="saliency", max_batches=args.saliency_batches)
    gxi_ch, gxi_ch_time = grad_saliency_per_channel(model, val_loader, device, method="gradxinput", max_batches=args.saliency_batches)

    s_rows = [{"channel": i, "saliency_mean": float(sal_ch[i])} for i in range(len(sal_ch))]
    write_csv(outdir / "saliency_channel_importance.csv", s_rows)

    g_rows = [{"channel": i, "gradxinput_mean": float(gxi_ch[i])} for i in range(len(gxi_ch))]
    write_csv(outdir / "grad_input_channel_importance.csv", g_rows)

    # Guardar matrices [C,T] en .npy por si luego quieres mapas temporales
    np.save(outdir / "saliency_channel_time.npy", sal_ch_time)   # shape [C,T]
    np.save(outdir / "gradxinput_channel_time.npy", gxi_ch_time) # shape [C,T]
    print("[EXPLAIN] saliency_channel_importance.csv, grad_input_channel_importance.csv, y mapas temporales .npy")

    # (4) Permutation importance por canal (val set)
    perm_rows = permutation_importance_channels(model, val_loader, device, repeats=args.perm_repeats, seed=42)
    write_csv(outdir / "permutation_importance_channels.csv", perm_rows)
    print("[EXPLAIN] permutation_importance_channels.csv")

    print(f"\n✓ Listo. Revisa la carpeta {outdir.resolve()}")
    print("  - Importancia por canal (weights/grad/gradxinput/perm) coherentes entre sí = evidencia fuerte.")
    print("  - Puedes cruzarlo con tus tests analíticos previos para el diagnóstico final.")

if __name__ == "__main__":
    main()

