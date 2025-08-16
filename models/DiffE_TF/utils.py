import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

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
# Loader 
# ------------------------------------------------------------------

def load_data_BCI(dataset_file: Path, subject_id: int = None):
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
        x = zscore_norm(torch.from_numpy(arr))
        x = pad_to_multiple(x, 8)
        return x

    def _prep_y(arr):
        return torch.from_numpy(arr).long()  # (T,)

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
        test_slice  = slice(i * 50,  (i + 1) * 50)

        X_train = X_train[train_slice]
        y_train = y_train[train_slice]
        X_val   = X_val[val_slice]
        y_val   = y_val[val_slice]
        X_test  = X_test[test_slice]
        y_test  = y_test[test_slice]
        print(f"[INFO] Seleccionando idx {train_slice} para X_train, {val_slice} para X_val, {test_slice} para X_test")

    else:
        print("[INFO] Cargando datos de todos los sujetos (concatenados)")

    print(f"[INFO] Dimensiones finales:")
    print(f"    - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"    - X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"    - X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_data_TOL(dataset_file: Path, subject_id: int = None, test_size=0.1, val_size=0.1, random_state=42):
    """
    Carga archivo .npz con X (trials, channels, samples), Y (trials,)
    subject_id: Si se pasa, selecciona solo los trials de ese sujeto (1-indexed)
    test_size: proporción para test split
    val_size: proporción para val split (sobre el resto tras test)
    """
    subject_trials = [200, 240, 180, 240, 240, 216, 240, 200, 240, 240]
    data = np.load(dataset_file)
    X = data['X']        # (trials, channels, samples)
    Y = data['Y']        # (trials,)

    if subject_id is not None:
        assert 1 <= subject_id <= len(subject_trials)
        start = sum(subject_trials[:subject_id-1])
        end = start + subject_trials[subject_id-1]
        X = X[start:end]
        Y = Y[start:end]
        print(f"[INFO] Seleccionando trials {start}:{end} para sujeto {subject_id}")
    else:
        print("[INFO] Usando trials de todos los sujetos (concatenados)")

    # Convertir a torch antes de normalizar
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).long()

    # Normalizar y pad
    X = zscore_norm(X)
    X = pad_to_multiple(X, mult=8)

    # Volver a numpy para usar sklearn splits
    X_np = X.numpy()
    Y_np = Y.numpy()

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, Y_np, test_size=test_size, random_state=random_state, stratify=Y_np
    )

    # Split train/val
    val_rel = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_rel, random_state=random_state, stratify=y_train
    )

    # Convertir de nuevo a tensor torch
    X_train = torch.from_numpy(X_train)
    X_val   = torch.from_numpy(X_val)
    X_test  = torch.from_numpy(X_test)
    y_train = torch.from_numpy(y_train)
    y_val   = torch.from_numpy(y_val)
    y_test  = torch.from_numpy(y_test)

    print(f"[INFO] Shapes finales:")
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
                              batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(EEGDataset(X_val,   y_val),
                              batch_size=batch_size_eval, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(EEGDataset(X_test,  y_test),
                              batch_size=batch_size_eval, shuffle=False, num_workers=4, pin_memory=True)
    
    print("[INFO] Número de batches:")
    print(f"    - Train: {len(train_loader)} batches")
    print(f"    - Val  : {len(val_loader)} batches")
    print(f"    - Test : {len(test_loader)} batches")

    return train_loader, val_loader, test_loader

