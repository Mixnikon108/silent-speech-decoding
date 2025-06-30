import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split   # ← sigue usándose en otros flujos

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
def load_npz_dataset(dataset_file: Path):
    """
    Carga el archivo .npz con claves
    X_train, y_train, X_val, y_val, X_test, y_test

    * X : (muestras, canales, trials)  →  (trials, canales, muestras)
    * y : one-hot (clases, trials)     →  (trials,) índices clase
    """
    data = np.load(dataset_file)

    def _prep_X(arr):
        arr = np.transpose(arr, (2, 1, 0)).astype(np.float32)       # (T, C, L)
        return zscore_norm(torch.from_numpy(arr))

    def _prep_y(arr):
        return torch.from_numpy(np.argmax(arr, axis=0)).long()      # (T,)

    X_train = _prep_X(data["X_train"])
    y_train = _prep_y(data["y_train"])
    X_val   = _prep_X(data["X_val"])
    y_val   = _prep_y(data["y_val"])
    X_test  = _prep_X(data["X_test"])
    y_test  = _prep_y(data["y_test"])

    return X_train, y_train, X_val, y_val, X_test, y_test


# ------------------------------------------------------------------
# NUEVO ❷ – creadores de DataLoader para las tres particiones
# ------------------------------------------------------------------
def get_dataloaders_npz(
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
