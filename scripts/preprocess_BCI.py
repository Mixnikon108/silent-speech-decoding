#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
import h5py
import mne
from scipy.stats import zscore, kurtosis

# ──────────────────────────── Rutas ────────────────────────────
project_dir   = Path(__file__).resolve().parent.parent        
raw_dir       = project_dir / "data" / "raw"       / "BCI2020"
processed_dir = project_dir / "data" / "processed" / "BCI2020"

# ───────────────────────────── Meta ────────────────────────────
subjects = [f"{i:02d}" for i in range(1, 16)]
classes  = ['hello', 'help-me', 'stop', 'thank-you', 'yes']

def get_channels():
    """
    Devuelve la lista de nombres de los canales EEG a partir del sujeto 01.
    """
    train_file = raw_dir / "training_set" / "data_sample01.mat"
    mat = sio.loadmat(train_file, squeeze_me=False, struct_as_record=False)

    # Extrae los nombres desde la estructura .clab del objeto epo_train
    channels = [ch.item() for ch in mat["epo_train"][0, 0].clab.flatten()]
    return channels

def _load_true_label_test(sid: str, path: Path):
    path_file = path / "test_set" / "track3_answer_sheet_test.xlsx"

    # Cargar etiquetas verdaderas del archivo Excel
    true_labels = (
        pd.read_excel(
            path_file,
            header=None,
            skiprows=2,
            engine="openpyxl"
        )
        .iloc[1:, 2::2]  
        .set_axis(
            [f"Data_Sample{i:02d}" for i in range(1, 16)],
            axis=1
        )
    )

    # Obtener columna de interés
    labels = true_labels[f"Data_Sample{sid}"].astype(int).values - 1 # Van de 0 a 5

    # Codificación one-hot
    num_classes = np.max(labels) + 1
    one_hot = np.eye(num_classes)[labels]

    return one_hot.T

def load_subject(sid: str):
    """Devuelve (x_train, y_train, x_val, y_val) de un sujeto."""

    train_file = raw_dir / "training_set"   / f"data_sample{sid}.mat"
    val_file   = raw_dir / "validation_set" / f"data_sample{sid}.mat"
    test_file  = raw_dir / "test_set"       / f"data_sample{sid}.mat"

    mat_tr = sio.loadmat(train_file, squeeze_me=False, struct_as_record=False)
    mat_va = sio.loadmat(val_file,   squeeze_me=False, struct_as_record=False)

    X_train = mat_tr["epo_train"][0, 0].x.astype(np.float32)
    y_train = mat_tr["epo_train"][0, 0].y.astype(np.uint8)

    X_val = mat_va["epo_validation"][0, 0].x.astype(np.float32)
    y_val = mat_va["epo_validation"][0, 0].y.astype(np.uint8)


    with h5py.File(test_file, "r") as f:
        grp = f["epo_test"]
        X_test = np.array(grp["x"], dtype=np.float32).T  
        y_test = _load_true_label_test(sid, raw_dir) # Son diferentes a los de entrenamiento y validación

    return X_train, y_train, X_val, y_val, X_test, y_test

def preprocess_subject(xt, yt, xv, yv, xts, yts):
    """Pre‑processes all epochs of a single subject following the Diff‑E pipeline.

    Steps
    -----
    1. Concatenate train/val/test along the trial dimension so CAR and ICA are
       computed on **all** trials (avoids bias caused by separate references).
    2. **Band‑pass 0.5–125 Hz** – removes DC drifts (<0.5 Hz) and keeps the
       high‑gamma band up to 125 Hz, matching Diff‑E. Implemented as a
       zero‑phase 4‑th‑order Butterworth IIR (`phase='zero'` applies forward
       and backward filtering to cancel group‑delay).
    3. **Notch at 60 Hz & 120 Hz** – suppresses mains interference (USA) and
       its first harmonic. Done with a 2‑nd‑order IIR notch in two passes
       because multiple stop‑bands aren’t allowed in a single IIR call.
    4. **Common Average Reference (CAR)** – re‑references signals to the global
       mean across 64 electrodes, mitigating spatial bias.
    5. **ICA artefact rejection** – FastICA with 25 components (≈40 % of
       channels, empirical sweet‑spot) and heuristic rejection based on
       variance & kurtosis (>3 SD). Removes EOG / EMG without EOG channels.
    6. **Baseline‑correction** – subtracts the mean of the −0.5 s window from
       each epoch so every trial starts centred in voltage.
    7. Splits `X_proc` back into train / val / test and returns them together
       with their unchanged labels.

    Parameters
    ----------
    xt, xv, xts : ndarray (samples, channels, trials)
        Raw EEG epochs for train, validation and test.
    yt, yv, yts : ndarray
        Corresponding labels (shape = (trials,)).

    Returns
    -------
    xt, yt, xv, yv, xts, yts : ndarray
        Pre‑processed splits with the same shapes as input.
    """

    # ------------------------------------------------------------------
    # 0) Concatenate along trials  → shape (trials, chan, samp)
    # ------------------------------------------------------------------
    X = np.concatenate([xt, xv, xts], axis=2).T  

    # ------------------------- parámetros --------------------------
    sfreq = 256                          # Hz
    bp_low, bp_high = 0.5, 125.0         # pasa-bandas
    notch_freqs = [60, 120]              # Hz
    t_baseline = 0.5                     # segundos (-500 ms)

    # ---------------------- 1) preparación -------------------------
    # MNE espera (n_epochs, n_channels, n_samples)
    info   = mne.create_info(get_channels(), sfreq, ch_types='eeg')
    info.set_montage('standard_1020')
    epochs = mne.EpochsArray(X, info, verbose=False)

    # ---------------------- 2) filtrado pasa-banda -----------------
    epochs.filter(bp_low, bp_high,
                method='iir',       # IIR avoids long FIR filters
                iir_params=dict(order=4, ftype='butter'),
                phase='zero',       # zero‑phase (forward+backward)      
                verbose=False)

    # ---------------------- 3) notch 60 / 120 ----------------------
    for f in notch_freqs:  # IIR notch ⇒ one stop‑band per call
        epochs._data = mne.filter.notch_filter(
            epochs._data,
            Fs=sfreq,
            freqs=f,                   
            method='iir',
            iir_params=dict(order=2, ftype='butter'),  # notch 2º orden
            phase='zero',              # ida-y-vuelta ⇒ fase lineal
            verbose=False,
        )

    # ---------------------- 4) CAR --------------------------------
    epochs.set_eeg_reference('average', verbose=False)

    # ---------------------- 5) ICA (artefactos EOG/EMG) -----------
    # Ajustamos ICA; 25 comp. suele ser bastante para 64 canales
    ica = mne.preprocessing.ICA(n_components=25, method='fastica', random_state=42, verbose=False)
    ica.fit(epochs)                              # aprende sobre los datos filtrados

    # Sin canales EOG específicos, marcamos automáticamente comp.
    # con varianza MUY alta o kurtosis extrema (proxy EOG/EMG)

    scores = zscore(np.log(np.var(ica.get_sources(epochs).get_data(), axis=2)))
    kurt   = zscore(kurtosis(ica.get_sources(epochs).get_data(), axis=2))

    reject = np.where((scores > 3) | (kurt > 3))[0]       # umbral heurístico
    ica.exclude = list(reject)
    epochs = ica.apply(epochs, exclude=ica.exclude, verbose=False)

    # ---------------------- 6) baseline correction ----------------
    # Intervalo de 0.5 s (128 muestras) al inicio de cada época
    bl_samples = int(t_baseline * sfreq)
    data = epochs.get_data()                               # (trials, chan, samp)

    baseline_mean = data[:, :, :bl_samples].mean(axis=2, keepdims=True)
    data -= baseline_mean

    # ---------------------- 7) Split back -----------------------------
    X_proc = data.astype(np.float32).T  # (samples, chan, trials)

    # Tamaños de cada partición original
    n_train = xt.shape[2]
    n_val   = xv.shape[2]
    n_test  = xts.shape[2]

    # Índices para separar en el array concatenado X_proc
    idx_val = n_train + n_val

    # División del array procesado X_proc en los splits originales
    xt   = X_proc[:,:,:n_train]              # train
    xv   = X_proc[:,:,n_train:idx_val]       # val
    xts  = X_proc[:,:,idx_val:]              # test

    return xt, yt, xv, yv, xts, yts

def load_all_subjects(apply_preprocessing : bool = False):
    """Carga todos los sujetos y devuelve los conjuntos de entrenamiento, validación y prueba."""
    X_train, y_train = [], []
    X_val,   y_val   = [], []
    X_test,  y_test  = [], []


    for sid in subjects:
        print(f"[{'dataset raw' if not apply_preprocessing else 'dataset filtered'}] Sujeto: {sid}/15")
        subject_data = load_subject(sid)  
 
        if apply_preprocessing:
            # Preprocesar los datos del sujeto
            subject_data = preprocess_subject(*subject_data)

        xt, yt, xv, yv, xts, yts = subject_data

        # Añadir datos del sujeto a las listas
        X_train.append(xt); y_train.append(yt)
        X_val.append(xv);   y_val.append(yv)
        X_test.append(xts); y_test.append(yts)

        print(f"Sujeto {sid}: " 
              f"X_train {xt.shape}, y_train {yt.shape}, "
              f"X_val {xv.shape}, y_val {yv.shape}, "
              f"X_test {xts.shape}, y_test {yts.shape}")
        
    # Concatenar todos los sujetos a lo largo de los trials
    X_train = np.concatenate(X_train, axis=2)
    y_train = np.concatenate(y_train, axis=1)

    X_val   = np.concatenate(X_val,   axis=2)
    y_val   = np.concatenate(y_val,   axis=1)

    X_test  = np.concatenate(X_test,  axis=2)
    y_test  = np.concatenate(y_test,  axis=1)


    def _prep_X(arr):
        x = np.transpose(arr, (2, 1, 0)).astype(np.float32)  # (T, C, L)
        return x

    def _prep_y(arr):
        return np.argmax(arr, axis=0)  # (T,)

    X_train = _prep_X(X_train)
    y_train = _prep_y(y_train)
    X_val   = _prep_X(X_val)
    y_val   = _prep_y(y_val)
    X_test  = _prep_X(X_test)
    y_test  = _prep_y(y_test)


    return X_train, y_train, X_val, y_val, X_test, y_test



# ────────────────────────── Carga del dataset ───────────────────────

print("Cargando raw dataset BCI2020...")

X_train, y_train, X_val, y_val, X_test, y_test = load_all_subjects()

print("\n Dimensiones del dataset raw:")
print(f"  - Train      : X = {X_train.shape}, y = {y_train.shape}")
print(f"  - Validation : X = {X_val.shape},  y = {y_val.shape}")
print(f"  - Test       : X = {X_test.shape},  y = {y_test.shape}")

raw_dataset_file = processed_dir / "BCI_raw.npz"

np.savez_compressed(raw_dataset_file,
                    X_train=X_train, y_train=y_train,
                    X_val=X_val,     y_val=y_val,
                    X_test=X_test,   y_test=y_test,
                    classes=np.array(classes, dtype=object))

print(f"\n✅ Dataset raw guardado en: {raw_dataset_file.relative_to(project_dir)}")


print("\nCargando dataset filtrado BCI2020...")

X_train_f, y_train_f, X_val_f, y_val_f, X_test_f, y_test_f = load_all_subjects(apply_preprocessing=True)

print("\n Dimensiones del dataset filtrado:")
print(f"  - Train      : X = {X_train.shape}, y = {y_train.shape}")
print(f"  - Validation : X = {X_val.shape},  y = {y_val.shape}")
print(f"  - Test       : X = {X_test.shape},  y = {y_test.shape}")

filtered_dataset_file = processed_dir / "BCI_processed.npz"

np.savez_compressed(filtered_dataset_file,
                    X_train=X_train_f, y_train=y_train_f,
                    X_val=X_val_f,     y_val=y_val_f,
                    X_test=X_test_f,   y_test=y_test_f,
                    classes=np.array(classes, dtype=object))

print(f"\n✅ Dataset filtrado guardado en: {filtered_dataset_file.relative_to(project_dir)}")








