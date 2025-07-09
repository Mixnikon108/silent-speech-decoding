#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
import h5py
import mne
from scipy.stats import zscore, kurtosis
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import mne

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
    notch_freq = 60                      # Hz
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

    # # ---------------------- 3) notch 60 / 120 ----------------------
    # for f in notch_freqs:  # IIR notch ⇒ one stop‑band per call
    #     epochs._data = mne.filter.notch_filter(
    #         epochs._data,
    #         Fs=sfreq,
    #         freqs=f,                   
    #         method='iir',
    #         iir_params=dict(order=2, ftype='butter'),  # notch 2º orden
    #         phase='zero',              # ida-y-vuelta ⇒ fase lineal
    #         verbose=False,
    #     )

    # # ---------------------- 4) CAR --------------------------------
    # epochs.set_eeg_reference('average', verbose=False)

    # # ---------------------- 5) ICA (artefactos EOG/EMG) -----------
    # # Ajustamos ICA; 25 comp. suele ser bastante para 64 canales
    # ica = mne.preprocessing.ICA(n_components=25, method='fastica', random_state=42, verbose=False)
    # ica.fit(epochs)                              # aprende sobre los datos filtrados

    # # Sin canales EOG específicos, marcamos automáticamente comp.
    # # con varianza MUY alta o kurtosis extrema (proxy EOG/EMG)

    # scores = zscore(np.log(np.var(ica.get_sources(epochs).get_data(), axis=2)))
    # kurt   = zscore(kurtosis(ica.get_sources(epochs).get_data(), axis=2))

    # reject = np.where((scores > 3) | (kurt > 3))[0]       # umbral heurístico
    # ica.exclude = list(reject)
    # epochs = ica.apply(epochs, exclude=ica.exclude, verbose=False)

    # # ---------------------- 6) baseline correction ----------------
    # # Intervalo de 0.5 s (128 muestras) al inicio de cada época
    # bl_samples = int(t_baseline * sfreq)
    # data = epochs.get_data()                               # (trials, chan, samp)

    # baseline_mean = data[:, :, :bl_samples].mean(axis=2, keepdims=True)
    # data -= baseline_mean

    # ---------------------- 7) Split back -----------------------------

    data = epochs.get_data() # >>>>> delete

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





    if data.shape[0] < data.shape[-1]:
        # (trials, chan, samples) (lo normal en MNE)
        data_fft = data
    else:
        # (samples, chan, trials) -> lo llevamos a (trials, chan, samples)
        data_fft = np.transpose(data, (2, 1, 0))

    print(data_fft.shape)
    N = data_fft.shape[2]        # nº muestras por epoch
    fs = sfreq
    freqs = np.fft.rfftfreq(N, 1/fs)

    # FFT: magnitud media en todos los canales y trials
    mag_fft = np.abs(np.fft.rfft(data_fft, axis=2)) / N       # (trials, chan, freqs)
    mean_fft = mag_fft.mean(axis=(0, 1))                      # media sobre trials y canales

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, mean_fft, label='Media FFT (trials+canales)', alpha=0.85)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud FFT")
    plt.title("Espectro medio tras preprocesado")
    plt.xlim(0, fs/2)
    plt.ylim(0, np.percentile(mean_fft, 95))  # Zoom automático sin el pico DC
    plt.axvline(bp_low, color='gray', ls='--', lw=0.8)
    plt.axvline(bp_high, color='gray', ls='--', lw=0.8)
    plt.tight_layout()
    plt.savefig("espectro_medio_preprocesado.png")
    plt.close()
    print("✅ Espectro medio guardado como 'espectro_medio_preprocesado.png'")
    return xt, yt, xv, yv, xts, yts


    # X = np.concatenate([xt, xv, xts], axis=2).T 
    # # ---------------------- parámetros ---------------------------
    # sfreq = 256
    # bp_low, bp_high = 0.5, 125.0
    # order = 4
    # t_baseline = 0.5

    # # Unidades: Asegúrate de trabajar en V para MNE, en µV para comparativa final
    # X_v = X * 1e-6  # Si X está en µV

    # # ---------------------- 1) MNE preparation ---------------------
    # info = mne.create_info(get_channels(), sfreq, ch_types='eeg')
    # info.set_montage('standard_1020')
    # epochs = mne.EpochsArray(X_v, info, verbose=False)

    # # ---------------------- 2) Filtro pasa-banda con MNE -----------
    # epochs.filter(bp_low, bp_high,
    #             method='iir',
    #             iir_params=dict(order=order, ftype='butter'),
    #             phase='zero',
    #             verbose=False)


    
    # # NOTCH
    # epochs._data = mne.filter.notch_filter(
    #     epochs._data,
    #     Fs=sfreq,
    #     freqs=60,  # Notch a 60 Hz             
    #     method='iir',
    #     iir_params=dict(order=order, ftype='butter'),  # notch 2º orden
    #     phase='zero',              # ida-y-vuelta ⇒ fase lineal
    #     verbose=False,
    # )

    # # CAR
    # epochs.set_eeg_reference('average', verbose=False)


    # # ICA
    # # Ajustamos ICA; 25 comp. suele ser bastante para 64 canales
    # ica = mne.preprocessing.ICA(n_components=25, method='fastica', random_state=42, verbose=False)
    # ica.fit(epochs)                              # aprende sobre los datos filtrados

    # # Sin canales EOG específicos, marcamos automáticamente comp.
    # # con varianza MUY alta o kurtosis extrema (proxy EOG/EMG)

    # scores = zscore(np.log(np.var(ica.get_sources(epochs).get_data(), axis=2)))
    # kurt   = zscore(kurtosis(ica.get_sources(epochs).get_data(), axis=2))

    # reject = np.where((scores > 3) | (kurt > 3))[0]       # umbral heurístico
    # ica.exclude = list(reject)
    # epochs = ica.apply(epochs, exclude=ica.exclude, verbose=False)

    # # Baseline correction
    # bl_samples = int(t_baseline * sfreq)
    # X_mne = epochs.get_data() * 1e6  # vuelve a µV para comparar

    # baseline_mean = X_mne[:, :, :bl_samples].mean(axis=2, keepdims=True)
    # X_mne -= baseline_mean

    

    # # ---------------------- 3) Filtro manual con SciPy -------------
    # # SciPy espera (trials, channels, samples), eje -1 es el de tiempo
    # b, a = butter(order, [bp_low, bp_high], btype='bandpass', fs=sfreq)
    # X_scipy = filtfilt(b, a, X, axis=2)  # filtra en samples

    # # ---------------------- 4) Gráfico comparativo -----------------
    # N = X.shape[2]
    # freqs = np.fft.rfftfreq(N, 1/sfreq)

    # # FFT sobre todos los trials/canales y medias
    # mag_raw   = np.abs(np.fft.rfft(X,       axis=2)) / N  # µV
    # mag_mne   = np.abs(np.fft.rfft(X_mne,   axis=2)) / N
    # mag_scipy = np.abs(np.fft.rfft(X_scipy, axis=2)) / N

    # mean_raw   = mag_raw.mean(axis=(0,1))
    # mean_mne   = mag_mne.mean(axis=(0,1))
    # mean_scipy = mag_scipy.mean(axis=(0,1))

    # canal_plot = 0  # Canal a visualizar (el primero)

    # plt.figure(figsize=(10, 5))
    # plt.plot(freqs, mag_raw[0].mean(axis=0),   label="Raw",          alpha=0.7)
    # plt.plot(freqs, mag_scipy[0].mean(axis=0), label="Filtro SciPy", alpha=0.8)
    # plt.plot(freqs, mag_mne[0].mean(axis=0),   label="Filtro MNE",   alpha=0.8, ls='--')
    # plt.axvline(bp_low,  color='gray', ls='--', lw=.8)
    # plt.axvline(bp_high, color='gray', ls='--', lw=.8)
    # plt.xlim(0, sfreq/2)
    # plt.ylim(0, np.percentile(mag_raw[0], 99))  # Zoom automático sin el pico DC (ajusta si quieres)
    # plt.xlabel("Frecuencia (Hz)")
    # plt.ylabel("Magnitud FFT (µV)")
    # plt.title("Comparativa filtrado: Raw vs SciPy vs MNE (canal 0, todos los trials)")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("comparativa_fft_canal0.png")
    # plt.close()
    # print("✅ Figura guardada como 'comparativa_fft_canal0.png'")


    # # ---------------------- 5) Split back --------------------------
    # X_proc = X_mne.astype(np.float32).transpose(2, 1, 0)  # (samples, channels, trials)
    # n_train = xt.shape[0]
    # n_val   = xv.shape[0]
    # n_test  = xts.shape[0]
    # idx_val = n_train + n_val

    # xt   = X_proc[:, :, :n_train]             # train
    # xv   = X_proc[:, :, n_train:idx_val]      # val
    # xts  = X_proc[:, :, idx_val:]             # test

    # return xt, yt, xv, yv, xts, yts

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

    return X_train, y_train, X_val, y_val, X_test, y_test



# ────────────────────────── Carga del dataset ───────────────────────

# print("Cargando raw dataset BCI2020...")

# X_train, y_train, X_val, y_val, X_test, y_test = load_all_subjects()

# print("\n Dimensiones del dataset raw:")
# print(f"  - Train      : X = {X_train.shape}, y = {y_train.shape}")
# print(f"  - Validation : X = {X_val.shape},  y = {y_val.shape}")
# print(f"  - Test       : X = {X_test.shape},  y = {y_test.shape}")

# raw_dataset_file = processed_dir / "raw_BCI2020.npz"

# np.savez_compressed(raw_dataset_file,
#                     X_train=X_train, y_train=y_train,
#                     X_val=X_val,     y_val=y_val,
#                     X_test=X_test,   y_test=y_test,
#                     classes=np.array(classes, dtype=object))

# print(f"\n✅ Dataset raw guardado en: {raw_dataset_file.relative_to(project_dir)}")

print("Cargando dataset filtrado BCI2020...")

X_train_f, y_train_f, X_val_f, y_val_f, X_test_f, y_test_f = load_all_subjects(apply_preprocessing=True)

print("\n Dimensiones del dataset filtrado:")
print(f"  - Train      : X = {X_train_f.shape}, y = {y_train_f.shape}")
print(f"  - Validation : X = {X_val_f.shape},  y = {y_val_f.shape}")
print(f"  - Test       : X = {X_test_f.shape},  y = {y_test_f.shape}")

filtered_dataset_file = processed_dir / "4_filtered_BCI2020.npz"

np.savez_compressed(filtered_dataset_file,
                    X_train=X_train_f, y_train=y_train_f,
                    X_val=X_val_f,     y_val=y_val_f,
                    X_test=X_test_f,   y_test=y_test_f,
                    classes=np.array(classes, dtype=object))

print(f"\n✅ Dataset filtrado guardado en: {filtered_dataset_file.relative_to(project_dir)}")








