"""
DIFFE_MOD
train.py — Lanzador de entrenamiento para Diff‑E (EEG imagined‑speech).

Ejemplo de uso:
    python train.py \
        --dataset_file data/processed/BCI2020/filtered_BCI2020.npz \
        --num_epochs 200 \
        --device cuda:0

El script centraliza la configuración vía CLI, simplifica las barras de
progreso y divide la lógica en funciones reutilizables para facilitar la
mantenimiento.


python main.py     --dataset_file C:/Users/jorge/Desktop/silent-speech-decoding/data/processed/BCI2020/filtered_BCI2020.npz    --device cpu     --num_epochs 1     --batch_train 250     --batch_eval 32     --seed 42     --alpha 0.1     --num_classes 5     --channels 64     --n_T 1000     --ddpm_dim 128     --encoder_dim 256     --fc_dim 512


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

from thop import profile 

# --- Repos propios ---------------------------------------------------------
from utils import load_data_BCI, load_data_TOL, get_dataloader  # función *.npz del proyecto
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
    print(f"[setup] Semilla fijada a {seed}")
    

def build_optim_and_sched(
    ddpm: nn.Module,
    diffe: nn.Module,
    *,
    base_lr: float = 3e-4,
    max_lr: float = 3e-3,
    step_size: int = 50,
    ) -> Tuple[Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler], ...]:
    """Devuelve parejas *(optimizer, scheduler)* para DDPM y Diff‑E."""
    print(f"[setup] Configurando optimizadores y schedulers: base_lr={base_lr}, max_lr={max_lr}, step_size={step_size}")

    opt_ddpm = optim.RMSprop(ddpm.parameters(), lr=base_lr * 0.5)
    opt_diffe = optim.RMSprop(diffe.parameters(), lr=base_lr * 2.0, weight_decay=1e-3)

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
    current_epoch: int = 0,
) -> None:
    """Ejecuta una época de entrenamiento completa."""
    ddpm.train()
    diffe.train()

    # Desempaquetado
    opt_ddpm, sched_ddpm = optimizers["ddpm"]
    opt_diffe, sched_diffe = optimizers["diffe"]
    crit_rec, crit_cls = criterions

    total_loss_ddpm = 0.0
    total_loss_gap = 0.0
    total_loss_cls = 0.0
    total_acc      = 0.0
    n_batches = 0

    warm = 5 # epochs de calentamiento del EMA

    for x, y in loaders["train"]:
        x = x.to(device)
        y = y.to(device, dtype=torch.long)
        # y_onehot = F.one_hot(y, num_classes=diffe.fc.linear_out[-1].out_features).float().to(device)

        # ------------------ DDPM ------------------
        opt_ddpm.zero_grad(set_to_none=True)
        x_hat, down, up, noise, t = ddpm(x)
        loss_ddpm = F.l1_loss(x_hat, x, reduction="none")
        l_ddpm = loss_ddpm.mean()
        l_ddpm.backward()
        opt_ddpm.step()
        sched_ddpm.step()

        # ------------------ Diff‑E ----------------
        opt_diffe.zero_grad(set_to_none=True)
        decoder_out, logits = diffe(x, (x_hat, down, up, t))
        loss_gap = crit_rec(decoder_out, loss_ddpm.detach())
        # loss_cls = crit_cls(logits, y_onehot)
        loss_cls = crit_cls(logits, y)
        loss = loss_gap + alpha * loss_cls
        loss.backward()
        opt_diffe.step()
        sched_diffe.step()

        # EMA update

        if current_epoch >= warm:
            ema_fc.update()
        else:
            # Evita el update durante el calentamiento
            ema_fc.copy_params_from_model_to_ema()

        # Acumula para promediar
        total_loss_ddpm += l_ddpm.item()
        total_loss_gap += loss_gap.item()
        total_loss_cls += loss_cls.item()

        with torch.no_grad():
            logits_train = diffe.fc(diffe.encoder(x)[1])
            preds = logits_train.argmax(dim=1)
            total_acc += (preds == y).float().mean().item()

        n_batches += 1
    train_acc = total_acc / n_batches
    return (total_loss_ddpm / n_batches,
            total_loss_gap / n_batches,
            total_loss_cls / n_batches,
            train_acc)


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrenamiento de Diff-E para imagined-speech EEG"
    )

    # --- Datos & dispositivo
    parser.add_argument("--dataset_file", type=str, required=True, help="Ruta al .npz pre‑procesado")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu", help="cpu | cuda:idx")

    # --- Hyper‑parámetros training
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_train", type=int, default=250)
    parser.add_argument("--batch_eval", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.1, help="Peso del término de clasificación")
    parser.add_argument("--subject_id", type=int, default=None)
    

    # --- Parámetros modelo
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--n_T", type=int, default=1000)
    parser.add_argument("--ddpm_dim", type=int, default=128)
    parser.add_argument("--encoder_dim", type=int, default=256)
    parser.add_argument("--fc_dim", type=int, default=512)

    # --- Experimentos & checkpoints
    parser.add_argument("--exp-dir", type=str, required=True, help="Directorio para guardar checkpoints y logs")

    args = parser.parse_args()
    print(f"[config] Parámetros: {args}")

    # ------------------ Seed & device ------------------
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"[setup] Usando dispositivo: {device}")

    # ------------------ Datos -------------------------
    dataset_file = Path(args.dataset_file)
    subject_id = args.subject_id

    if args.dataset == "BCI":
        print("[data] Cargando datos del dataset BCI")
        X_train, y_train, X_val, y_val, X_test, y_test = load_data_BCI(dataset_file=dataset_file, subject_id=subject_id)
    elif args.dataset == "TOL":
        print("[data] Cargando datos del dataset TOL")
        X_train, y_train, X_val, y_val, X_test, y_test = load_data_TOL(dataset_file=dataset_file, subject_id=subject_id)

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
    print("[model] Inicializando modelos...")
    ddpm_model = ConditionalUNet(in_channels=args.channels, 
                                 n_feat=args.ddpm_dim).to(device)
    
    ddpm = DDPM(nn_model=ddpm_model, 
                betas=(1e-6, 1e-2), 
                n_T=args.n_T, 
                device=device).to(device)

    encoder = Encoder(in_channels=args.channels, 
                      dim=args.encoder_dim).to(device)
    
    decoder = Decoder(in_channels=args.channels, 
                      n_feat=args.ddpm_dim, 
                      encoder_dim=args.encoder_dim).to(device)
    
    fc = LinearClassifier(args.encoder_dim, args.fc_dim, emb_dim=args.num_classes).to(device)

    diffe = DiffE(encoder, decoder, fc).to(device)


    # ------------------ Info FLOPs & tamaño ----------
    total_params = sum(p.numel() for p in ddpm.parameters()) + sum(p.numel() for p in diffe.parameters())
    model_size_mb = total_params * 4 / (1024 ** 2)  # asumiendo float32
   
    try:
        dummy_x = torch.randn(1, args.channels, X_train.shape[-1]).to(device)
        flops_ddpm, _ = profile(ddpm, inputs=(dummy_x,), verbose=False)
        flops_enc, _ = profile(diffe.encoder, inputs=(dummy_x,), verbose=False)
        flops_total = flops_ddpm + flops_enc
        flops_str = f"{flops_total / 1e9:.2f} GFLOPs"
    except Exception as err:  
        flops_str = f"N/A (error: {err})"

    print(f"[model] FLOPs: {flops_str} | Peso: {model_size_mb:.2f} MB | Parámetros: {total_params/1e6:.2f} M")


    # ------------------ Optimizadores / EMA ----------
    (opt_ddpm, sched_ddpm), (opt_diffe, sched_diffe) = build_optim_and_sched(ddpm, diffe)
    ema_fc = EMA(diffe.fc, beta=0.95, update_after_step=10, update_every=5)
    print("[setup] EMA configurada en el clasificatorio (beta=0.95)")

    criterions = (nn.L1Loss(), nn.CrossEntropyLoss())

    # ------------------ Directorio de experimento ----
    exp_dir = Path(args.exp_dir)
    best_ckpt_path = exp_dir / "best.pt"
    last_ckpt_path = exp_dir / "last.pt"

    # ------------------ Training loop ----------------
    best_metrics: Dict[str, float] = {}
    print(f"[train] Inicio del bucle de entrenamiento por {args.num_epochs} épocas")

    pbar = tqdm(
        range(args.num_epochs),
        ncols=88,
        desc="Overall",
        unit="epoch",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]"
    )

    
    for epoch in pbar:
        print(f"\n[train] ===== Época {epoch+1}/{args.num_epochs} =====")
        train_loss_ddpm, train_loss_gap, train_loss_cls, train_acc  = train_epoch(
            ddpm=ddpm,
            diffe=diffe,
            loaders=loaders,
            criterions=criterions,
            optimizers={"ddpm": (opt_ddpm, sched_ddpm), "diffe": (opt_diffe, sched_diffe)},
            ema_fc=ema_fc,
            device=device,
            alpha=args.alpha,
            current_epoch=epoch
        )
        
        print(f"[train] acc: {train_acc:.3f} | Loss DDPM: {train_loss_ddpm:.5f} | Loss Gap: {train_loss_gap:.5f} | Loss Cls: {train_loss_cls:.5f}")

        # --- Evaluación ---
        ddpm.eval()
        diffe.eval()
        with torch.no_grad():
            metrics_full = evaluate(
                diffe.encoder,
                ema_fc,
                loaders["val"],
                device,
                num_classes=args.num_classes,
            )
        metrics = metrics_full["metrics"]
        acc = metrics["accuracy"]
        f1 = metrics["macro_f1"]
        print(f"[eval] Epoch {epoch+1} — acc: {acc:.4f}, macro_f1: {f1:.4f}")


        # --- Checkpointing --------------------------------------------------
        torch.save(
            {
                "epoch": epoch + 1,
                "ddpm_state": ddpm.state_dict(),
                "diffe_state": diffe.state_dict(),
                "opt_ddpm_state": opt_ddpm.state_dict(),
                "opt_diffe_state": opt_diffe.state_dict(),
                "metrics": metrics,
            },
            last_ckpt_path,
        )

        if acc > best_metrics.get("accuracy", 0.0):
            best_metrics = metrics
            torch.save(
                {
                    "epoch": epoch + 1,
                    "ddpm_state": ddpm.state_dict(),
                    "diffe_state": diffe.state_dict(),
                    "opt_ddpm_state": opt_ddpm.state_dict(),
                    "opt_diffe_state": opt_diffe.state_dict(),
                    "metrics": metrics,
                },
                best_ckpt_path,
            )
            print(f"[checkpoint] ¡Nueva mejor accuracy: {acc:.4f}! Checkpoint guardado en {best_ckpt_path}")

        pbar.set_postfix(best_acc=best_metrics.get("accuracy", 0.0))

    # ------------------ Resumen final ----------------
    print("\n[test] Evaluando el mejor modelo en el conjunto de test…")
    # Cargamos el mejor estado para asegurar coherencia
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        ddpm.load_state_dict(ckpt["ddpm_state"])
        diffe.load_state_dict(ckpt["diffe_state"])
        print(f"[test] Checkpoint restaurado desde {best_ckpt_path}")

    ddpm.eval()
    diffe.eval()
    with torch.no_grad():
        test_report = evaluate(
            diffe.encoder,
            ema_fc,
            loaders["test"],
            device,
            num_classes=args.num_classes,
            return_cm=True,
        )

    print("[test] Reporte completo:")
    for section, values in test_report.items():
        print(f"\n  {section}:")
        if isinstance(values, dict):
            for k, v in values.items():
                # Si es escalar (Python float/int o NumPy scalar), formateamos; si es array, lo mostramos tal cual
                if isinstance(v, (float, int, np.floating, np.integer)):
                    print(f"    {k:>16s}: {float(v):.4f}")
                else:  # p. ej. ndarray => matriz de confusión
                    print(f"    {k:>16s}: {v}")
        else:  # values es ndarray (matriz de confusión principal)
            print(values)





if __name__ == "__main__":
    main()
