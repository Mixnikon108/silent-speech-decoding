import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ------------------------------------------------------------------
# Padding hasta múltiplo de N (8 en nuestro caso)
# ------------------------------------------------------------------
def pad_to_multiple(x: torch.Tensor, mult: int = 8):
    """
    Devuelve x padded por la derecha para que su última dimensión
    (tiempo) sea múltiplo de `mult`. No modifica x in-place.
    Se hace multiplo de 8 para que sea compatible con la convolución causal
    y evitar problemas de padding en la convolución.
    """
    L = x.shape[-1]
    pad = (mult - L % mult) % mult   # 0..mult-1
    if pad:
        print(f"[INFO] Padding aplicado: se añaden {pad} valores a la derecha (dim original: {L})")
        x = F.pad(x, (0, pad), mode="reflect")  # (left, right)
    else:
        print(f"[INFO] No se aplica padding (dimensión temporal ya es múltiplo de {mult}): {L}")
    print(f"[INFO] Dimensión temporal final: {x.shape[-1]}")
    return x


# ------------------------------------------------------------------
# Normalización trial-a-trial (z-score sobre canales y muestras)
# ------------------------------------------------------------------
def zscore_norm(data: torch.Tensor) -> torch.Tensor:
    """
    Aplica z-score por trial: se normaliza cada trial (a lo largo de todos sus canales y muestras).
    """
    print(f"[INFO] Aplicando z-score normalization por trial")
    print(f"    - Forma de entrada (esperada): (trials, canales, muestras) = {data.shape}")

    mean = torch.mean(data, dim=(1, 2), keepdim=True)
    std  = torch.std(data, dim=(1, 2), keepdim=True)

    print(f"    - Calculando media y desviación estándar por trial")
    print(f"    - Mean shape: {mean.shape}, Std shape: {std.shape}")

    normalized = (data - mean) / (std + 1e-8)

    print(f"    - Normalización completada. Forma de salida: {normalized.shape}")
    return normalized


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
# Loader para filtered_BCI2020.npz
# ------------------------------------------------------------------
def load_data(dataset_file: Path, subject_id: int = None):
    """
    Carga el archivo .npz con claves
    X_train, y_train, X_val, y_val, X_test, y_test

    * X : (muestras, canales, trials)  →  (trials, canales, muestras)
    * y : one-hot (clases, trials)     →  (trials,) índices clase

    Si se pasa un subject_id (1-15), se devuelven solo los datos de ese sujeto.
    """
    print(f"[INFO] Cargando datos desde: {dataset_file}")
    data = np.load(dataset_file)

    def _prep_X(arr):
        arr = np.transpose(arr, (2, 1, 0)).astype(np.float32)  # (T, C, L)
        x = zscore_norm(torch.from_numpy(arr))
        x = pad_to_multiple(x, 8)
        return x

    def _prep_y(arr):
        return torch.from_numpy(np.argmax(arr, axis=0)).long()  # (T,)

    X_train = _prep_X(data["X_train"])
    y_train = _prep_y(data["y_train"])
    X_val   = _prep_X(data["X_val"])
    y_val   = _prep_y(data["y_val"])
    X_test  = _prep_X(data["X_test"])
    y_test  = _prep_y(data["y_test"])

    if subject_id is not None:
        assert 1 <= subject_id <= 15, "subject_id must be between 1 and 15"
        i = subject_id - 1
        print(f"[INFO] Filtrando datos para el sujeto {subject_id} (índice interno: {i})")

        train_slice = slice(i * 300, (i + 1) * 300)
        val_slice   = slice(i * 50,  (i + 1) * 50)
        test_slice  = slice((i + 1) * 50, (i + 2) * 50)

        X_train = X_train[train_slice]
        y_train = y_train[train_slice]
        X_val   = X_val[val_slice]
        y_val   = y_val[val_slice]
        X_test  = X_test[test_slice]
        y_test  = y_test[test_slice]

    else:
        print("[INFO] Cargando datos de todos los sujetos (concatenados)")

    print(f"[INFO] Dimensiones finales:")
    print(f"    - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"    - X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"    - X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test



# ------------------------------------------------------------------
# NUEVO ❷ – creadores de DataLoader para las tres particiones
# ------------------------------------------------------------------
def get_dataloader(
    X_train, y_train, X_val, y_val, X_test, y_test,
    batch_size: int = 32, batch_size_eval: int = 256, shuffle: bool = True
):

    print("[INFO] Creando DataLoaders:")
    print(f"    - Batch size (train): {batch_size}  | Shuffle: {shuffle}")
    print(f"    - Batch size (eval) : {batch_size_eval}  | Shuffle: False")    

    train_loader = DataLoader(EEGDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=shuffle)
    val_loader   = DataLoader(EEGDataset(X_val,   y_val),
                              batch_size=batch_size_eval, shuffle=False)
    test_loader  = DataLoader(EEGDataset(X_test,  y_test),
                              batch_size=batch_size_eval, shuffle=False)
    
    print("[INFO] Número de batches:")
    print(f"    - Train: {len(train_loader)} batches")
    print(f"    - Val  : {len(val_loader)} batches")
    print(f"    - Test : {len(test_loader)} batches")

    return train_loader, val_loader, test_loader

# project_dir = Path(__file__).resolve().parent.parent.parent   
# dataset_file = project_dir / "data" / "processed" / "BCI2020"  / "filtered_BCI2020.npz"

# print(f"Dataset file: {dataset_file}")  