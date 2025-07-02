"""
train.py — Lanzador de entrenamiento para Diff‑E (EEG imagined‑speech).

Ejemplo de uso:
    python train.py \
        --dataset_file data/processed/BCI2020/filtered_BCI2020.npz \
        --num_epochs 200 \
        --device cuda:0

El script centraliza la configuración vía CLI, simplifica las barras de
progreso y divide la lógica en funciones reutilizables para facilitar la
mantenimiento.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ema_pytorch import EMA
from tqdm.auto import tqdm

# --- Repos propios ---------------------------------------------------------
from utils import load_data, get_dataloader  # función *.npz del proyecto
from models import (
    ConditionalUNet,
    DDPM,
    DiffE,
    Encoder,
    Decoder,
    LinearClassifier,
)
from evaluation import evaluate  # versión extendida con baseline aleatorio

# ---------------------------------------------------------------------------
# Utilidades auxiliares
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Fija *todas* las seeds para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CuDNN determinista ‑ necesario para reproducir
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optim_and_sched(
    ddpm: nn.Module,
    diffe: nn.Module,
    *,
    base_lr: float = 9e-5,
    max_lr: float = 1.5e-3,
    step_size: int = 150,
    ) -> Tuple[Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler], ...]:
    """Devuelve parejas *(optimizer, scheduler)* para DDPM y Diff‑E."""
    opt_ddpm = optim.RMSprop(ddpm.parameters(), lr=base_lr)
    opt_diffe = optim.RMSprop(diffe.parameters(), lr=base_lr)

    sched_ddpm = optim.lr_scheduler.CyclicLR(
        opt_ddpm,
        base_lr=base_lr,
        max_lr=max_lr,
        step_size_up=step_size,
        cycle_momentum=False,
        mode="exp_range",
        gamma=0.9998,
    )
    sched_diffe = optim.lr_scheduler.CyclicLR(
        opt_diffe,
        base_lr=base_lr,
        max_lr=max_lr,
        step_size_up=step_size,
        cycle_momentum=False,
        mode="exp_range",
        gamma=0.9998,
    )
    return (opt_ddpm, sched_ddpm), (opt_diffe, sched_diffe)


def train_epoch(
    *,
    ddpm: nn.Module,
    diffe: DiffE,
    loaders: Dict[str, torch.utils.data.DataLoader],
    criterions: Tuple[nn.Module, nn.Module],
    optimizers: Dict[str, Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]],
    ema_fc: EMA,
    device: torch.device,
    alpha: float,
) -> None:
    """Ejecuta una época de entrenamiento completa."""

    ddpm.train()
    diffe.train()

    # Desempaquetado
    opt_ddpm, sched_ddpm = optimizers["ddpm"]
    opt_diffe, sched_diffe = optimizers["diffe"]
    crit_rec, crit_cls = criterions

    for x, y in loaders["train"]:
        x = x.to(device)
        y = y.to(device, dtype=torch.long)
        y_onehot = F.one_hot(y, num_classes=diffe.fc.linear_out[-1].out_features).float().to(device)

        # ------------------ DDPM ------------------
        opt_ddpm.zero_grad(set_to_none=True)
        x_hat, down, up, noise, t = ddpm(x)
        loss_ddpm = crit_rec(x_hat, x)
        loss_ddpm.backward()
        opt_ddpm.step()
        sched_ddpm.step()

        # ------------------ Diff‑E ----------------
        opt_diffe.zero_grad(set_to_none=True)
        decoder_out, logits = diffe(x, (x_hat, down, up, t))
        loss_gap = crit_rec(decoder_out, loss_ddpm.detach())
        loss_cls = crit_cls(logits, y_onehot)
        (loss_gap + alpha * loss_cls).backward()
        opt_diffe.step()
        sched_diffe.step()

        ema_fc.update()


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrenamiento de Diff‑E para imagined‑speech EEG"
    )

    # --- Datos & dispositivo
    parser.add_argument("--dataset_file", type=str, required=True, help="Ruta al .npz pre‑procesado")
    parser.add_argument("--device", type=str, default="cuda:0", help="cpu | cuda:idx")

    # --- Hyper‑parámetros training
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_train", type=int, default=250)
    parser.add_argument("--batch_eval", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.1, help="Peso del término de clasificación")

    # --- Parámetros modelo
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--n_T", type=int, default=1000)
    parser.add_argument("--ddpm_dim", type=int, default=128)
    parser.add_argument("--encoder_dim", type=int, default=256)
    parser.add_argument("--fc_dim", type=int, default=512)

    args = parser.parse_args()

    # ------------------ Seed & device ------------------
    set_seed(args.seed)
    device = torch.device(args.device)

    # ------------------ Datos -------------------------
    dataset_file = Path(args.dataset_file)
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(dataset_file=dataset_file)

    loaders = {}
    loaders["train"], loaders["val"], loaders["test"] = get_dataloader(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        args.batch_train,
        args.batch_eval,
        shuffle=True,
    )

    # ------------------ Modelos -----------------------
    ddpm_model = ConditionalUNet(in_channels=args.channels, n_feat=args.ddpm_dim).to(device)
    ddpm = DDPM(nn_model=ddpm_model, betas=(1e-6, 1e-2), n_T=args.n_T, device=device).to(device)

    encoder = Encoder(in_channels=args.channels, dim=args.encoder_dim).to(device)
    decoder = Decoder(in_channels=args.channels, n_feat=args.ddpm_dim, encoder_dim=args.encoder_dim).to(device)
    fc = LinearClassifier(args.encoder_dim, args.fc_dim, emb_dim=args.num_classes).to(device)

    diffe = DiffE(encoder, decoder, fc).to(device)

    # ------------------ Optimizadores / EMA ----------
    (opt_ddpm, sched_ddpm), (opt_diffe, sched_diffe) = build_optim_and_sched(ddpm, diffe)
    ema_fc = EMA(diffe.fc, beta=0.95, update_after_step=100, update_every=10)

    criterions = (nn.L1Loss(), nn.MSELoss())

    # ------------------ Training loop ----------------
    best_metrics: Dict[str, float] = {}

    pbar = tqdm(
        range(args.num_epochs),
        ncols=88,
        bar_format="{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, best acc: {postfix[best_acc]:.2%}]",
    )

    for epoch in pbar:
        train_epoch(
            ddpm=ddpm,
            diffe=diffe,
            loaders=loaders,
            criterions=criterions,
            optimizers={"ddpm": (opt_ddpm, sched_ddpm), "diffe": (opt_diffe, sched_diffe)},
            ema_fc=ema_fc,
            device=device,
            alpha=args.alpha,
        )

        # --- Evaluación ---
        ddpm.eval()
        diffe.eval()
        with torch.no_grad():
            metrics_full = evaluate(
                diffe.encoder,
                ema_fc,
                loaders["test"],
                device,
                num_classes=args.num_classes,
            )
        metrics = metrics_full["metrics"]  # nos quedamos con las métricas primarias

        acc = metrics["accuracy"]
        if acc > best_metrics.get("accuracy", 0.0):
            best_metrics = metrics

        pbar.set_postfix(best_acc=best_metrics.get("accuracy", 0.0))

    # ------------------ Resumen final ----------------
    print("\nMejores métricas alcanzadas:")
    for k, v in best_metrics.items():
        print(f"  {k:>16s}: {v:.4f}")


if __name__ == "__main__":
    main()
