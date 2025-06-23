#!/usr/bin/env python3
"""
Create a mixed-subject dataset for BCI Competition 2020 – Track 3.

Resultado:
    processed/raw/BCI2020/bcic2020_mixed.npz
    ├─ X_train : float32 (total_trials_train, 64, 1000)
    ├─ y_train : uint8   (total_trials_train,)
    ├─ X_val   : float32 (total_trials_val,   64, 1000)
    ├─ y_val   : uint8   (total_trials_val,)
    └─ classes : <object> ['hello', 'help-me', 'stop', 'thank-you', 'yes']
"""

from pathlib import Path
import numpy as np
import scipy.io as sio

# ──────────────────────────── Rutas ────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent        # silent-speech-decoding
RAW_DIR      = PROJECT_DIR / "data" / "raw" / "BCI2020"
OUT_DIR      = PROJECT_DIR / "processed" / "raw" / "BCI2020"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ───────────────────────────── Meta ────────────────────────────
SUBJECTS = [f"{i:02d}" for i in range(1, 16)]
CLASSES  = ['hello', 'help-me', 'stop', 'thank-you', 'yes']

# ────────────────────── Utilidades de carga ────────────────────
def load_subject(sid: str):
    """Devuelve (x_train, y_train, x_val, y_val) de un sujeto."""
    train_file = RAW_DIR / "Training set"   / f"Data_Sample{sid}.mat"
    val_file   = RAW_DIR / "Validation set" / f"Data_Sample{sid}.mat"

    mat_tr = sio.loadmat(train_file, squeeze_me=True)
    mat_va = sio.loadmat(val_file,   squeeze_me=True)

    # Datos y etiquetas (one-hot → entero)
    x_tr = mat_tr["epo_train"]["x"].astype(np.float32)       # (C, T, N)
    y_tr = mat_tr["epo_train"]["y"].argmax(0).astype(np.uint8)
    x_va = mat_va["epo_validation"]["x"].astype(np.float32)
    y_va = mat_va["epo_validation"]["y"].argmax(0).astype(np.uint8)

    # Reordenar a (N, C, T) y pad temporal hasta 1000
    x_tr = np.transpose(x_tr, (2, 0, 1))
    x_va = np.transpose(x_va, (2, 0, 1))
    x_tr = np.pad(x_tr, ((0, 0), (0, 0), (0, 5)), mode="edge")
    x_va = np.pad(x_va, ((0, 0), (0, 0), (0, 5)), mode="edge")

    return x_tr, y_tr, x_va, y_va

# ────────────────────────── Construcción ───────────────────────
if __name__ == "__main__":
    X_tr, Y_tr, X_va, Y_va = [], [], [], []

    for sid in SUBJECTS:
        xt, yt, xv, yv = load_subject(sid)
        X_tr.append(xt); Y_tr.append(yt)
        X_va.append(xv); Y_va.append(yv)
        print(f"Sujeto {sid}: train {xt.shape}, val {xv.shape}")

    X_train = np.concatenate(X_tr, axis=0)
    y_train = np.concatenate(Y_tr, axis=0)
    X_val   = np.concatenate(X_va, axis=0)
    y_val   = np.concatenate(Y_va, axis=0)

    out_file = OUT_DIR / "bcic2020_mixed.npz"
    np.savez_compressed(out_file,
                        X_train=X_train, y_train=y_train,
                        X_val=X_val,     y_val=y_val,
                        classes=np.array(CLASSES, dtype=object))
    print(f"\n✅ Dataset guardado en: {out_file.relative_to(PROJECT_DIR)}")
