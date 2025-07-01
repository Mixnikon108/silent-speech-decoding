import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split   # ← sigue usándose en otros flujos
import torch.nn.functional as F

import torch.nn.functional as F

# ------------------------------------------------------------------
# Padding hasta múltiplo de N (8 en nuestro caso)
# ------------------------------------------------------------------
def pad_to_multiple(x: torch.Tensor, mult: int = 8):
    """
    Devuelve x padded por la derecha para que su última dimensión
    (tiempo) sea múltiplo de `mult`. No modifica x in-place.
    """
    L = x.shape[-1]
    pad = (mult - L % mult) % mult   # 0..mult-1
    if pad:
        x = F.pad(x, (0, pad), mode="reflect")  # (left, right)
    return x


# TMPDIR=$(pwd) pip install torch==2.2.2+cpu   --no-cache-dir   -f https://download.pytorch.org/whl/torch_stable.html

# ------------------------------------------------------------------
# Normalización trial-a-trial (z-score sobre canales y muestras)
# ------------------------------------------------------------------
def zscore_norm(data: torch.Tensor) -> torch.Tensor:
    mean = torch.mean(data, dim=(1, 2), keepdim=True)
    std  = torch.std(data, dim=(1, 2), keepdim=True)
    return (data - mean) / (std + 1e-8)


# ------------------------------------------------------------------
# Dataset genérico
# ------------------------------------------------------------------
class EEGDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ------------------------------------------------------------------
# NUEVO ❶  –  cargador para filtered_BCI2020.npz
# ------------------------------------------------------------------
def load_data(dataset_file: Path):
    """
    Carga el archivo .npz con claves
    X_train, y_train, X_val, y_val, X_test, y_test

    * X : (muestras, canales, trials)  →  (trials, canales, muestras)
    * y : one-hot (clases, trials)     →  (trials,) índices clase
    """
    data = np.load(dataset_file)

    def _prep_X(arr):
        arr = np.transpose(arr, (2, 1, 0)).astype(np.float32)       # (T, C, L)
        x = zscore_norm(torch.from_numpy(arr))
        x = pad_to_multiple(x, 8)
        return x

    def _prep_y(arr):
        return torch.from_numpy(np.argmax(arr, axis=0)).long()      # (T,)

    X_train = _prep_X(data["X_train_f"])
    y_train = _prep_y(data["y_train_f"])
    X_val   = _prep_X(data["X_val_f"])
    y_val   = _prep_y(data["y_val_f"])
    X_test  = _prep_X(data["X_test_f"])
    y_test  = _prep_y(data["y_test_f"])

    return X_train, y_train, X_val, y_val, X_test, y_test


# ------------------------------------------------------------------
# NUEVO ❷ – creadores de DataLoader para las tres particiones
# ------------------------------------------------------------------
def get_dataloader(
    X_train, y_train, X_val, y_val, X_test, y_test,
    batch_size: int = 32, batch_size_eval: int = 256, shuffle: bool = True
):
    train_loader = DataLoader(EEGDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=shuffle)
    val_loader   = DataLoader(EEGDataset(X_val,   y_val),
                              batch_size=batch_size_eval, shuffle=False)
    test_loader  = DataLoader(EEGDataset(X_test,  y_test),
                              batch_size=batch_size_eval, shuffle=False)
    return train_loader, val_loader, test_loader


# project_dir = Path(__file__).resolve().parent.parent.parent   
# dataset_file = project_dir / "data" / "processed" / "BCI2020"  / "filtered_BCI2020.npz"
