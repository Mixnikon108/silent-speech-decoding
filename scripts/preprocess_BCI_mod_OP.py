#!/usr/bin/env python3

from pathlib import Path
import argparse
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
    channels = [ch.item() for ch in mat["epo_train"][0, 0].clab.flatten()]
    return channels

def _load_true_label_test(sid: str, path: Path):
    path_file = path / "test_set" / "track3_answer_sheet_test.xlsx"
    true_labels = (
        pd.read_excel(
            path_file,
            header=None,
            skiprows=2,
            engine="openpyxl"
        )
        .iloc[1:, 2::2]
        .set_axis([f"Data_Sample{i:02d}" for i in range(1, 16)], axis=1)
    )
    labels = true_labels[f"Data_Sample{sid}"].astype(int).values - 1
    num_classes = np.max(labels) + 1
    one_hot = np.eye(num_classes)[labels]
    return one_hot.T

def load_subject(sid: str):
    """Devuelve (x_train, y_train, x_val, y_val, x_test, y_test) de un sujeto."""
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
        y_test = _load_true_label_test(sid, raw_dir)

    return X_train, y_train, X_val, y_val, X_test, y_test

def preprocess_subject(
    xt, yt, xv, yv, xts, yts,
    *,
    # ------ pasa-banda ------
    enable_bandpass=True,
    l_freq=0.05,
    h_freq=None,
    # ------ notch mains -----
    enable_notch=True,
    # ------ CAR -------------
    enable_car=True,
    # ------ ICA -------------
    enable_ica=False,
    # ------ baseline --------
    enable_baseline=True,
    # ------ salida ----------
    residual=False
):
    """
    Preprocesa (opcionalmente) y devuelve los splits con el mismo formato de entrada.
    Shapes de entrada esperadas: (n_channels, n_samples, n_trials)
    Devuelve: xt_proc, yt, xv_proc, yv, xts_proc, yts
    Si residual=True, devuelve R = X_raw - X_filtrado.
    """
    
    # ---- PRINT DEL PIPELINE ----
    print("\n=== Preprocesamiento configurado ===")
    print(f"Bandpass: {enable_bandpass} | l_freq={l_freq}, h_freq={h_freq}")
    print(f"Notch (60Hz): {enable_notch}")
    print(f"CAR: {enable_car}")
    print(f"ICA: {enable_ica}")
    print(f"Baseline: {enable_baseline}")
    print(f"Residual: {residual}")
    print("===================================\n")

    sfreq = 256.0
    montage = 'standard_1020'

    # 0) Concatenar → (trials, chan, samp)
    X = np.concatenate([xt, xv, xts], axis=2).transpose(2, 0, 1)
    X_raw = X.copy()

    # preparar MNE
    info = mne.create_info(ch_names=get_channels(), sfreq=sfreq, ch_types='eeg')
    info.set_montage(montage)
    epochs = mne.EpochsArray(X, info, verbose=False)

    # 1) pasa-banda
    if enable_bandpass:
        epochs.filter(
            l_freq=l_freq, h_freq=h_freq,
            method='iir',
            iir_params=dict(order=4, ftype='butter'),
            phase='zero',
            verbose=False
        )

    # 2) notch (60 Hz)
    if enable_notch:
        epochs._data = mne.filter.notch_filter(
            epochs._data,
            Fs=256.0,
            freqs=60,
            method='iir',
            iir_params=dict(order=2, ftype='butter'),
            phase='zero',
            verbose=False
        )

    # 3) CAR
    if enable_car:
        epochs.set_eeg_reference('average', verbose=False)

    # 4) ICA (heurístico)
    if enable_ica:
        ica = mne.preprocessing.ICA(
            n_components=25,
            method='fastica',
            random_state=42,
            verbose=False
        )
        ica.fit(epochs)
        src = ica.get_sources(epochs).get_data()  # (trials, comps, samp)
        var_scores  = zscore(np.log(np.var(src, axis=2)))
        kurt_scores = zscore(kurtosis(src, axis=2, fisher=True, bias=False))
        bad_var = np.where(var_scores.mean(axis=0) > 3.0)[0]
        bad_kur = np.where(kurt_scores.mean(axis=0) > 3.0)[0]
        ica.exclude = list(sorted(set(bad_var.tolist() + bad_kur.tolist())))
        epochs = ica.apply(epochs, exclude=ica.exclude, verbose=False)

    # 5) baseline
    if enable_baseline:
        bl_samples = int(0.5 * 256.0)
        data = epochs.get_data()
        if 0 < bl_samples <= data.shape[2]:
            baseline_mean = data[:, :, :bl_samples].mean(axis=2, keepdims=True)
            data = data - baseline_mean
            epochs._data = data

    # 6) salida
    X_filt = epochs.get_data().astype(np.float32)
    data_out = (X_raw - X_filt).astype(np.float32) if residual else X_filt

    # 7) reconstruir splits
    X_proc = data_out.transpose(2, 1, 0)  # (samples, chan, trials)
    n_train = xt.shape[2]
    n_val   = xv.shape[2]
    idx_val = n_train + n_val
    xt_out  = X_proc[:, :, :n_train]
    xv_out  = X_proc[:, :, n_train:idx_val]
    xts_out = X_proc[:, :, idx_val:]
    return xt_out, yt, xv_out, yv, xts_out, yts

def load_all_subjects(apply_preprocessing: bool = False, **preproc_kwargs):
    """Carga todos los sujetos y devuelve los conjuntos de entrenamiento, validación y prueba."""
    X_train, y_train = [], []
    X_val,   y_val   = [], []
    X_test,  y_test  = [], []

    mode = "raw"
    if apply_preprocessing and preproc_kwargs.get("residual", False):
        mode = "residual"
    elif apply_preprocessing:
        mode = "filtered"

    for sid in subjects:
        print(f"[dataset {mode}] Sujeto: {sid}/15")
        subject_data = load_subject(sid)
        if apply_preprocessing:
            subject_data = preprocess_subject(*subject_data, **preproc_kwargs)

        xt, yt, xv, yv, xts, yts = subject_data
        X_train.append(xt); y_train.append(yt)
        X_val.append(xv);   y_val.append(yv)
        X_test.append(xts); y_test.append(yts)

        print(f"Sujeto {sid}: "
              f"X_train {xt.shape}, y_train {yt.shape}, "
              f"X_val {xv.shape}, y_val {yv.shape}, "
              f"X_test {xts.shape}, y_test {yts.shape}")

    # Concatenar por trials
    X_train = np.concatenate(X_train, axis=2); y_train = np.concatenate(y_train, axis=1)
    X_val   = np.concatenate(X_val,   axis=2); y_val   = np.concatenate(y_val,   axis=1)
    X_test  = np.concatenate(X_test,  axis=2); y_test  = np.concatenate(y_test,  axis=1)

    def _prep_X(arr):
        return np.transpose(arr, (2, 1, 0)).astype(np.float32)  # (T, C, L)
    def _prep_y(arr):
        return np.argmax(arr, axis=0)  # (T,)

    X_train = _prep_X(X_train); y_train = _prep_y(y_train)
    X_val   = _prep_X(X_val);   y_val   = _prep_y(y_val)
    X_test  = _prep_X(X_test);  y_test  = _prep_y(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test

# ──────────────────────────── CLI ──────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Construye el dataset BCI2020 (raw/filtered/residual) con opciones de preprocesado."
    )
    p.add_argument("-o", "--output", type=str, required=True,
                   help="Nombre del archivo .npz de salida (relativo a data/processed/BCI2020 o ruta absoluta).")
    # Activar/desactivar preprocesado global
    g = p.add_mutually_exclusive_group()
    g.add_argument("--preprocess", dest="preprocess", action="store_true",  help="Aplicar preprocesado.")
    g.add_argument("--no-preprocess", dest="preprocess", action="store_false", help="No aplicar preprocesado.")
    p.set_defaults(preprocess=True)

    # Límites del paso-banda
    p.add_argument("--l-freq", type=float, default=0.05, help="Frecuencia de corte inferior (Hz) del filtro pasa-banda.")
    p.add_argument("--h-freq", type=float, default=None, help="Frecuencia de corte superior (Hz) del filtro pasa-banda (None para solo high-pass).")

    # Flags de pasos
    bpg = p.add_mutually_exclusive_group()
    bpg.add_argument("--bandpass", dest="enable_bandpass", action="store_true", help="Habilitar paso-banda.")
    bpg.add_argument("--no-bandpass", dest="enable_bandpass", action="store_false", help="Deshabilitar paso-banda.")
    p.set_defaults(enable_bandpass=True)

    ng = p.add_mutually_exclusive_group()
    ng.add_argument("--notch", dest="enable_notch", action="store_true", help="Habilitar notch 60 Hz.")
    ng.add_argument("--no-notch", dest="enable_notch", action="store_false", help="Deshabilitar notch.")
    p.set_defaults(enable_notch=True)

    carg = p.add_mutually_exclusive_group()
    carg.add_argument("--car", dest="enable_car", action="store_true", help="Habilitar referencia promedio (CAR).")
    carg.add_argument("--no-car", dest="enable_car", action="store_false", help="Deshabilitar CAR.")
    p.set_defaults(enable_car=True)

    icag = p.add_mutually_exclusive_group()
    icag.add_argument("--ica", dest="enable_ica", action="store_true", help="Habilitar ICA (heurístico).")
    icag.add_argument("--no-ica", dest="enable_ica", action="store_false", help="Deshabilitar ICA.")
    p.set_defaults(enable_ica=False)

    blg = p.add_mutually_exclusive_group()
    blg.add_argument("--baseline", dest="enable_baseline", action="store_true", help="Habilitar corrección de baseline.")
    blg.add_argument("--no-baseline", dest="enable_baseline", action="store_false", help="Deshabilitar baseline.")
    p.set_defaults(enable_baseline=True)

    p.add_argument("--residual", action="store_true", help="En vez de X filtrado, guardar residual R = X_raw - X_filtrado.")

    return p.parse_args()

def main():
    args = parse_args()

    # Coherencia residual/preprocess
    if args.residual and not args.preprocess:
        print("[AVISO] --residual requiere preprocesado; activando --preprocess.")
        args.preprocess = True

    # Cargar dataset (con o sin preprocesado)
    X_train, y_train, X_val, y_val, X_test, y_test = load_all_subjects(
        apply_preprocessing=args.preprocess,
        enable_bandpass=args.enable_bandpass,
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        enable_notch=args.enable_notch,
        enable_car=args.enable_car,
        enable_ica=args.enable_ica,
        enable_baseline=args.enable_baseline,
        residual=args.residual
    )

    print("\nDimensiones del dataset:")
    print(f"  - Train      : X = {X_train.shape}, y = {y_train.shape}")
    print(f"  - Validation : X = {X_val.shape},  y = {y_val.shape}")
    print(f"  - Test       : X = {X_test.shape},  y = {y_test.shape}")

    # Resolver ruta de salida
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = processed_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        classes=np.array(classes, dtype=object)
    )

    rel = out_path.relative_to(project_dir) if str(out_path).startswith(str(project_dir)) else out_path
    print(f"\n✅ Dataset guardado en: {rel}")

if __name__ == "__main__":
    main()
