# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# from utils import load_data_BCI  # Asegúrate de tener esta función


# file = 'LP005'  # Cambia a 'RAW' o 'BP' según necesites

# # Cargar datos RAW
# raw_file = Path("/home/w314/w314139/PROJECT/silent-speech-decoding/data/processed/BCI2020/BCI_raw.npz")
# X_train, y_train, X_val, y_val, X_test, y_test = load_data_BCI(raw_file, subject_id=1)

# # Cargar datos PROCESADOS
# processed_file = Path(f"/home/w314/w314139/PROJECT/silent-speech-decoding/data/processed/BCI2020/BCI_{file}.npz")
# X_train_proc, y_trainf, X_valf, y_valf, X_testf, y_testf = load_data_BCI(processed_file, subject_id=1)


# # Selección del primer trial
# x_raw = X_train[0]          # Shape: (64, 795)
# x_proc = X_train_proc[0]    # Shape asumida igual: (64, 795)

# # FFT por canal
# fft_raw = np.fft.fft(x_raw, axis=-1)
# fft_proc = np.fft.fft(x_proc, axis=-1)

# # Magnitud y promedio sobre canales
# mag_raw = np.abs(fft_raw)
# mag_proc = np.abs(fft_proc)
# avg_spectrum_raw = np.mean(mag_raw, axis=0)
# avg_spectrum_proc = np.mean(mag_proc, axis=0)

# # Frecuencias
# fs = 256  # Ajusta si necesario
# freqs = np.fft.fftfreq(x_raw.shape[1], d=1/fs)

# # # Usar solo la mitad positiva del espectro
# # half = x_raw.shape[1] // 2
# # freqs = freqs[:half]
# # avg_spectrum_raw = avg_spectrum_raw[:half]
# # avg_spectrum_proc = avg_spectrum_proc[:half]
# mask = (freqs >= 0) & (freqs <= 10)
# freqs = freqs[mask]
# avg_spectrum_raw = avg_spectrum_raw[mask]
# avg_spectrum_proc = avg_spectrum_proc[mask]

# # Plot (sin mostrar, solo guardar)
# plt.figure(figsize=(10, 5))
# plt.plot(freqs, avg_spectrum_raw, label='Raw', alpha=0.8)
# plt.plot(freqs, avg_spectrum_proc, label='Processed', alpha=0.8)
# plt.title('Espectro de Frecuencia - Primer Trial')
# plt.xlabel('Frecuencia (Hz)')
# plt.ylabel('Magnitud')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

# # Guardar figura
# output_path = Path(f"./spectrum_comparison_{file}.png")  # Cambia la ruta si quieres otro destino
# plt.savefig(output_path, dpi=300)
# plt.close()




# # Obtener la señal del primer canal del primer trial
# signal_raw = X_train[0, 0, :]      # Canal 0, trial 0
# signal_proc = X_train_proc[0, 0, :]  # Mismo canal y trial

# # Eje de tiempo
# fs = 256  # Frecuencia de muestreo (ajusta si necesario)
# t = np.arange(signal_raw.shape[0]) / fs  # en segundos

# # Crear la figura
# plt.figure(figsize=(10, 5))
# plt.plot(t, signal_raw, label='Raw', alpha=0.7)
# plt.plot(t, signal_proc, label='Processed', alpha=0.7)
# plt.title('Señal del Primer Canal - Primer Trial')
# plt.xlabel('Tiempo (s)')
# plt.ylabel('Amplitud')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

# # Guardar figura
# output_path_time = Path(f"./signal_comparison_channel0_trial0_{file}.png")
# plt.savefig(output_path_time, dpi=300)
# plt.close()




import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import load_data_TOL  # Asegúrate de tener esta función

# Cargar datos RAW
raw_file = Path("/home/w314/w314139/PROJECT/silent-speech-decoding/data/processed/TOL/TOL_raw.npz")
X_train, y_train, X_val, y_val, X_test, y_test = load_data_TOL(raw_file, subject_id=1)

# Cargar datos PROCESADOS
processed_file = Path("/home/w314/w314139/PROJECT/silent-speech-decoding/data/processed/TOL/TOL_LP005.npz")
X_train_proc, y_trainf, X_valf, y_valf, X_testf, y_testf = load_data_TOL(processed_file, subject_id=1)


# Selección del primer trial
x_raw = X_train[0]          # Shape: (64, 795)
x_proc = X_train_proc[0]    # Shape asumida igual: (64, 795)

# FFT por canal
fft_raw = np.fft.fft(x_raw, axis=-1)
fft_proc = np.fft.fft(x_proc, axis=-1)

# Magnitud y promedio sobre canales
mag_raw = np.abs(fft_raw)
mag_proc = np.abs(fft_proc)
avg_spectrum_raw = np.mean(mag_raw, axis=0)
avg_spectrum_proc = np.mean(mag_proc, axis=0)

# Frecuencias
fs = 256  # Ajusta si necesario
freqs = np.fft.fftfreq(x_raw.shape[1], d=1/fs)

# Usar solo la mitad positiva del espectro
half = x_raw.shape[1] // 2
freqs = freqs[:half]
avg_spectrum_raw = avg_spectrum_raw[:half]
avg_spectrum_proc = avg_spectrum_proc[:half]

# Plot (sin mostrar, solo guardar)
plt.figure(figsize=(10, 5))
plt.plot(freqs, avg_spectrum_raw, label='Raw', alpha=0.8)
plt.plot(freqs, avg_spectrum_proc, label='Processed', alpha=0.8)
plt.title('Espectro de Frecuencia - Primer Trial')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Guardar figura
output_path = Path("./spectrum_comparison_TOL.png")  # Cambia la ruta si quieres otro destino
plt.savefig(output_path, dpi=300)
plt.close()


# Obtener la señal del primer canal del primer trial
signal_raw = X_train[0, 0, :]      # Canal 0, trial 0
signal_proc = X_train_proc[0, 0, :]  # Mismo canal y trial

# Eje de tiempo
fs = 256  # Frecuencia de muestreo (ajusta si necesario)
t = np.arange(signal_raw.shape[0]) / fs  # en segundos

# Crear la figura
plt.figure(figsize=(10, 5))
plt.plot(t, signal_raw, label='Raw', alpha=0.7)
plt.plot(t, signal_proc, label='Processed', alpha=0.7)
plt.title('Señal del Primer Canal - Primer Trial')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Guardar figura
output_path_time = Path("./signal_comparison_channel0_trial0_TOL.png")
plt.savefig(output_path_time, dpi=300)
plt.close()















# import numpy as np
# from pathlib import Path
# from collections import Counter, defaultdict

# # Estadística / ML
# from scipy.stats import kruskal
# from sklearn.feature_selection import mutual_info_classif
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import StratifiedKFold, cross_val_score
# from statsmodels.stats.multitest import multipletests

# # Utilidades
# import json, os

# # ==== Configuración ====
# DATA_ROOT = Path("/home/w314/w314139/PROJECT/silent-speech-decoding/data/processed/BCI2020")
# RAW_FILE = DATA_ROOT / "BCI_raw.npz"       # Cambia si procede
# PROC_FILE = DATA_ROOT / "BCI_LP005.npz"    # Opcional, no se usa para tests analíticos
# SUBJECT_ID = 1
# FS = 256
# LOW_BANDS = [(0.0,1.0),(0.5,4.0),(4.0,8.0)]  # ULF/delta/theta de ejemplo
# N_MI_BINS = 16
# N_PERMUTATIONS = 200  # para tests de permutación (ajusta si quieres más potencia)
# CV_FOLDS = 5
# RANDOM_STATE = 42
# OUTDIR = Path("./analysis_outputs")
# OUTDIR.mkdir(parents=True, exist_ok=True)

# # ==== Carga ====
# from utils import load_data_BCI  # Debes tener esta función
# X_train, y_train, X_val, y_val, X_test, y_test = load_data_BCI(RAW_FILE, subject_id=SUBJECT_ID)

# # Asumimos shapes: X_*: (n_trials, n_channels, T); y_*: (n_trials,)
# def to_numpy_int(y):
#     y = np.asarray(y)
#     if y.ndim>1:
#         y = y.squeeze()
#     # garantizar entero [0..K-1]
#     classes, y_enc = np.unique(y, return_inverse=True)
#     return y_enc, classes

# y_train_enc, classes = to_numpy_int(y_train)
# y_val_enc, _ = to_numpy_int(y_val)
# y_test_enc, _ = to_numpy_int(y_test)
# n_classes = len(classes)

# n_tr, n_ch, T = X_train.shape
# print(f"Train shape: {X_train.shape}; classes={n_classes}, chans={n_ch}, T={T}")

# # ==== Helpers ====
# def bandpower_fft(x, fs, f_lo, f_hi):
#     # x: (..., T)
#     Xf = np.fft.rfft(x, axis=-1)
#     freqs = np.fft.rfftfreq(x.shape[-1], d=1/fs)
#     m = (freqs >= f_lo) & (freqs < f_hi)
#     # potencia = suma |X|^2 / N (sin normalizar por ventana; consistente para comparar)
#     p = (np.abs(Xf[..., m])**2).sum(axis=-1)
#     return p

# def slope_and_offset(x):
#     # Ajuste lineal por mínimos cuadrados por canal y trial: x_t = a*t + b
#     # Devuelve (a,b)
#     t = np.arange(x.shape[-1])
#     t = (t - t.mean()) / t.std()  # normalizar t para estabilidad
#     # a = cov(x,t)/var(t), b = mean(x) (porque t centrado)
#     # Broadcast: x: (..., T), t: (T,)
#     xm = x.mean(axis=-1, keepdims=True)
#     cov = ((x - xm) * t).sum(axis=-1)
#     var_t = (t**2).sum()
#     a = cov / var_t
#     b = xm.squeeze(-1)
#     return a, b

# def fdr(pvals, alpha=0.05, method="fdr_bh"):
#     rej, p_corr, _, _ = multipletests(pvals, alpha=alpha, method=method)
#     return rej, p_corr

# def eta_squared_from_kw(H, k, N):
#     # Aproximación de tamaño de efecto para Kruskal–Wallis
#     # Referencia común: eta^2_KW = (H - k + 1) / (N - k)
#     return max(0.0, (H - k + 1.0) / (N - k)) if N>k else np.nan

# def vectorize_trials(X):
#     # Aplana (n_trials, n_channels, T) -> (n_trials, n_channels*T)
#     n, c, t = X.shape
#     return X.reshape(n, c*t)

# def cv_score_model(X, y, clf):
#     cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
#     scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=None)
#     return scores.mean(), scores.std()

# rng = np.random.default_rng(RANDOM_STATE)

# # ==== (1) Importancia de electrodos: tests por canal ====
# # Features por canal y trial: offset, slope, bandpowers bajas
# a, b = slope_and_offset(X_train)  # a: (n_trials, n_channels), b igual
# features = defaultdict(list)
# pvals_kw = []
# etas = []
# channels = list(range(n_ch))

# # construir matriz features_ch: (n_trials, n_features) para cada canal, y hacer KW contra y
# feat_names = [f"BP_{lo:.1f}-{hi:.1f}Hz" for (lo,hi) in LOW_BANDS] + ["slope","offset"]
# per_channel_results = []

# for ch in channels:
#     feats_ch = []
#     # bandpowers
#     for (lo,hi) in LOW_BANDS:
#         bp = bandpower_fft(X_train[:, ch, :], FS, lo, hi)
#         feats_ch.append(bp)
#     # slope & offset
#     feats_ch.append(a[:, ch])
#     feats_ch.append(b[:, ch])
#     feats_ch = np.vstack(feats_ch).T  # (n_trials, n_features)
#     # Kruskal–Wallis univar por feature y Stouffer combine p-values o min-p?
#     # Usamos min-p: más conservador para señal.
#     Hs, ps = [], []
#     for fi in range(feats_ch.shape[1]):
#         groups = [feats_ch[y_train_enc==cls, fi] for cls in range(n_classes)]
#         H, p = kruskal(*groups)
#         Hs.append(H); ps.append(p)
#     p_min = np.min(ps)
#     pvals_kw.append(p_min)
#     # tamaño de efecto aproximado usando H más grande
#     H_max = float(np.max(Hs))
#     eta = eta_squared_from_kw(H_max, n_classes, len(y_train_enc))
#     etas.append(eta)

#     # MI multivariada (discretizando con quantiles internos de sklearn)
#     # Para estabilidad, estandarizamos feats_ch
#     Xf = StandardScaler().fit_transform(feats_ch)
#     mi = mutual_info_classif(Xf, y_train_enc, discrete_features=False, random_state=RANDOM_STATE)
#     mi_sum = float(np.sum(mi))

#     per_channel_results.append({
#         "channel": ch,
#         "kw_min_p": float(p_min),
#         "kw_eta2_from_maxH": eta,
#         "mi_sum": mi_sum,
#         **{f"MI_{fn}": float(miv) for fn, miv in zip(feat_names, mi)}
#     })

# rej_kw, p_kw_corr = fdr(pvals_kw, alpha=0.05)
# for i, d in enumerate(per_channel_results):
#     d["kw_p_adj"] = float(p_kw_corr[i])
#     d["kw_significant"] = bool(rej_kw[i])

# # Guardar CSV
# import pandas as pd
# df_ch = pd.DataFrame(per_channel_results).sort_values(
#     by=["kw_significant","kw_p_adj","mi_sum","kw_eta2_from_maxH"],
#     ascending=[False, True, False, False]
# )
# df_ch.to_csv(OUTDIR / "channel_importance_stats.csv", index=False)

# print("\nTop canales por evidencia (Kruskal FDR + MI):")
# print(df_ch.head(10)[["channel","kw_p_adj","kw_significant","kw_eta2_from_maxH","mi_sum"]])

# # ==== (1b) Ablaciones de identidad de canal con un clasificador sencillo ====
# def evaluate_with_channel_permutation(X, y, mode="none"):
#     """
#     mode:
#       - "none": sin permutar
#       - "fixed": una permutación fija de canales igual para todos los trials
#       - "per_trial": permutación aleatoria independiente por trial (destruye identidad)
#     """
#     Xp = X.copy()
#     if mode == "fixed":
#         perm = rng.permutation(X.shape[1])
#         Xp = Xp[:, perm, :]
#     elif mode == "per_trial":
#         for i in range(X.shape[0]):
#             perm = rng.permutation(X.shape[1])
#             Xp[i] = Xp[i, perm, :]

#     Xv = vectorize_trials(Xp)
#     Xv = StandardScaler(with_mean=True, with_std=True).fit_transform(Xv)
#     clf = LogisticRegression(
#         penalty="l2", solver="saga", C=1.0, max_iter=2000, random_state=RANDOM_STATE, n_jobs=None
#     )
#     mu, sd = cv_score_model(Xv, y, clf)
#     return mu, sd

# abl_results = []
# for mode in ["none","fixed","per_trial"]:
#     mu, sd = evaluate_with_channel_permutation(X_train, y_train_enc, mode=mode)
#     abl_results.append({"mode": mode, "cv_acc_mean": mu, "cv_acc_std": sd})
# df_abl = pd.DataFrame(abl_results)
# df_abl.to_csv(OUTDIR / "channel_identity_ablation.csv", index=False)
# print("\nAblaciones de identidad de canal (CV accuracy):")
# print(df_abl)

# # Interpretación:
# # - none vs fixed ~ igual → el índice del canal no importa por sí mismo (el modelo es indiferente a reordenar consistentemente).
# # - none >> per_trial → dependencia fuerte de identidad del canal a nivel de trial.

# # ==== (2) Señal temporal / posibles bloques ====
# # Asumimos que X_train está en orden “temporal” original (como vienen los trials).
# # Si no lo está, sustituye por la secuencia original.

# y_seq = y_train_enc.copy()
# n = len(y_seq)

# # (2a) Autocorrelación por lags y test por permutación
# def label_autocorr(y, max_lag=10):
#     # codificamos clases a enteros 0..K-1 y centramos
#     y_num = y.astype(float)
#     y_num = (y_num - y_num.mean()) / (y_num.std() + 1e-9)
#     ac = []
#     for lag in range(1, max_lag+1):
#         v = y_num[:-lag] * y_num[lag:]
#         ac.append(float(v.mean()))
#     return np.array(ac)  # shape (max_lag,)

# def perm_pvalue_stat(y, stat_fn, n_perm=N_PERMUTATIONS, rng=None):
#     rng = rng or np.random.default_rng()
#     observed = stat_fn(y)
#     count = 0
#     for _ in range(n_perm):
#         yp = rng.permutation(y)
#         val = stat_fn(yp)
#         if abs(val) >= abs(observed):
#             count += 1
#     p = (count + 1) / (n_perm + 1)
#     return observed, p

# ac = label_autocorr(y_seq, max_lag=10)
# # p-valor para lag-1 usando permutación
# obs_lag1, p_lag1 = perm_pvalue_stat(y_seq, lambda y: label_autocorr(y, max_lag=1)[0], n_perm=N_PERMUTATIONS, rng=rng)

# # (2b) Runs test (Wald–Wolfowitz) para 2+ clases: reducimos a binario por “clase mayoritaria vs resto”
# maj = Counter(y_seq).most_common(1)[0][0]
# y_bin = (y_seq == maj).astype(int)

# def runs_test_binary(x):
#     # x: 0/1
#     runs = 1 + np.sum(x[1:] != x[:-1])
#     n1 = x.sum()
#     n2 = len(x) - n1
#     if n1 == 0 or n2 == 0:
#         return {"runs": runs, "z": np.nan, "p": 1.0}
#     mu = 1 + 2*n1*n2/(n1+n2)
#     var = (2*n1*n2*(2*n1*n2 - n1 - n2)) / (((n1+n2)**2) * (n1+n2-1))
#     z = (runs - mu) / np.sqrt(var + 1e-9)
#     # p-valor bilateral aproximando normal
#     from scipy.stats import norm
#     p = 2*(1 - norm.cdf(abs(z)))
#     return {"runs": int(runs), "z": float(z), "p": float(p), "n1": int(n1), "n2": int(n2)}

# runs_res = runs_test_binary(y_bin)

# # (2c) MI entre índice temporal y etiqueta
# t_idx = np.arange(n).reshape(-1,1)
# mi_time = mutual_info_classif(t_idx, y_seq, discrete_features=True, random_state=RANDOM_STATE)
# # Permutación para p-valor
# def mi_stat(y):
#     return float(mutual_info_classif(t_idx, y, discrete_features=True, random_state=RANDOM_STATE)[0])
# obs_mi, p_mi = perm_pvalue_stat(y_seq, mi_stat, n_perm=N_PERMUTATIONS, rng=rng)

# # (2d) Clasificador con solo “tiempo”
# def eval_time_only_classifier(y, n_bins=10):
#     # features: [t_norm, bin_index (one-hot)]
#     t_norm = (t_idx - t_idx.mean()) / (t_idx.std() + 1e-9)
#     bins = np.digitize(t_idx.flatten(), np.linspace(0, n-1, n_bins+1)[1:-1])
#     X_feat = np.hstack([t_norm, bins.reshape(-1,1)])
#     scaler = StandardScaler().fit(X_feat)
#     Xs = scaler.transform(X_feat)
#     clf = LogisticRegression(
#         penalty="l2", solver="lbfgs", multi_class="auto",
#         max_iter=2000, random_state=RANDOM_STATE
#     )
#     mu, sd = cv_score_model(Xs, y, clf)
#     return mu, sd

# time_only_mu, time_only_sd = eval_time_only_classifier(y_seq, n_bins=min(10, max(3, n//20)))

# # (2e) CUSUM sobre proporción de clase mayoritaria + permutación
# def cusum_binary(x):
#     # x: 0/1, desviación de la proporción global
#     p_hat = x.mean()
#     s = np.cumsum(x - p_hat)
#     # estadístico: rango máx-min
#     return float(s.max() - s.min())

# obs_cusum, p_cusum = perm_pvalue_stat(y_bin, cusum_binary, n_perm=N_PERMUTATIONS, rng=rng)

# # ==== Guardar resultados temporales ====
# summary_temporal = {
#     "autocorr_lag1": float(obs_lag1),
#     "autocorr_lag1_p": float(p_lag1),
#     "autocorr_lags_1to10": [float(v) for v in ac],
#     "runs_test": runs_res,
#     "MI_time": float(obs_mi),
#     "MI_time_p": float(p_mi),
#     "time_only_classifier_cv_acc_mean": float(time_only_mu),
#     "time_only_classifier_cv_acc_std": float(time_only_sd),
#     "cusum_binary_stat": float(obs_cusum),
#     "cusum_binary_p": float(p_cusum),
#     "n_trials": int(n),
#     "n_classes": int(n_classes),
#     "majority_class": int(maj),
# }

# with open(OUTDIR / "temporal_structure_summary.json", "w") as f:
#     json.dump(summary_temporal, f, indent=2)

# print("\n=== RESULTADOS TEMPORALES (resumen) ===")
# for k,v in summary_temporal.items():
#     if isinstance(v, (float,int,str)):
#         print(f"{k}: {v}")
# print("Detalle completo en temporal_structure_summary.json")

# # ==== Interpretación rápida impresa ====
# def chance_level(c):
#     return 1.0/c if c>0 else np.nan

# print("\n=== INTERPRETACIÓN RÁPIDA ===")
# print("- Si muchos canales tienen kw_p_adj < 0.05 y MI alta → hay señal discriminativa asociada a canales concretos.")
# print("- En ablation: si acc(none) ≈ acc(fixed) pero acc(per_trial) cae → el modelo depende de la identidad de canal.")
# print(f"- Time-only clf acc ≈ {time_only_mu:.3f} (azar ≈ {chance_level(n_classes):.3f}). Si >> azar → hay patrón temporal explotable.")
# print(f"- Autocorr lag1 = {obs_lag1:.4f} (p={p_lag1:.3f}); MI(time)={obs_mi:.4f} (p={p_mi:.3f}); CUSUM p={p_cusum:.3f}.")
# print("- Runs test p<0.05 sugiere no-aleatoriedad (posibles bloques).")
# print(f"\nCSV guardados en: {OUTDIR.resolve()}")


# -----------------------------------------------------------------------------------------------------







# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# from utils import load_data_BCI
# from scipy.signal import butter, filtfilt
# from scipy.stats import f_oneway, kruskal
# import csv

# # ----------------------- Parámetros -----------------------
# SUBJECT_ID   = 1
# FS           = 256.0            # Hz
# UL_CUTOFF_HZ = 0.1              # Banda ultra-baja (low-pass)
# OUT_DIR      = Path("./ultra_low_analysis")
# OUT_DIR.mkdir(parents=True, exist_ok=True)

# # Rutas de datos (ajústalas si cambian)
# RAW_FILE = Path("/home/w314/w314139/PROJECT/silent-speech-decoding/data/processed/BCI2020/BCI_raw.npz")

# # ----------------------- Utilidades -----------------------
# def butter_lowpass(data, cutoff_hz, fs, order=4):
#     """Low-pass IIR Butterworth + filtfilt por época y canal."""
#     if cutoff_hz is None:
#         return data
#     nyq = 0.5 * fs
#     Wn  = cutoff_hz / nyq
#     b, a = butter(order, Wn, btype="low", analog=False)
#     # data: (n_trials, n_channels, n_samples)
#     y = np.empty_like(data, dtype=np.float32)
#     for i in range(data.shape[0]):
#         y[i] = filtfilt(b, a, data[i], axis=-1).astype(np.float32)
#     return y

# def compute_offset_slope(X, fs):
#     """
#     X: (n_trials, n_channels, n_samples)
#     return:
#       offset_ch: (n_trials, n_channels) media temporal por canal
#       slope_ch:  (n_trials, n_channels) pendiente por canal (OLS rápido)
#       offset_g:  (n_trials,) media sobre canales
#       slope_g:   (n_trials,) media sobre canales
#     """
#     n_tr, n_ch, n_s = X.shape
#     t = (np.arange(n_s, dtype=np.float32) / fs)
#     t_mean = t.mean()
#     t_var  = float(((t - t_mean)**2).sum()) + 1e-12

#     offset_ch = X.mean(axis=2)  # (n_tr, n_ch)

#     slope_ch = np.empty((n_tr, n_ch), dtype=np.float32)
#     for i in range(n_tr):
#         Xi = X[i]                               # (n_ch, n_s)
#         x_mean = Xi.mean(axis=1, keepdims=True) # (n_ch, 1)
#         cov_tx = ((t - t_mean) * (Xi - x_mean)).sum(axis=1)  # (n_ch,)
#         slope_ch[i] = (cov_tx / t_var).astype(np.float32)

#     offset_g = offset_ch.mean(axis=1)
#     slope_g  = slope_ch.mean(axis=1)
#     return offset_ch, slope_ch, offset_g, slope_g

# def ul_power(X_ul):
#     """Potencia (varianza) de la señal filtrada < UL_CUTOFF_HZ por trial/canal y global."""
#     pow_ch = X_ul.var(axis=2)        # (n_trials, n_channels)
#     pow_g  = pow_ch.mean(axis=1)     # (n_trials,)
#     return pow_ch, pow_g

# def per_class_lists(vec, y):
#     """Convierte vec (n_trials,) en lista de arrays por clase (para tests/boxplots)."""
#     classes = np.unique(y)
#     return [vec[y == c] for c in classes], classes

# def save_boxplot(data_per_class, classes, title, ylabel, out_png):
#     labels = [f"Cls {int(c)} (n={len(data_per_class[i])})" for i, c in enumerate(classes)]
#     plt.figure(figsize=(10, 5))
#     plt.boxplot(data_per_class, labels=labels, showmeans=True)
#     plt.title(title)
#     plt.ylabel(ylabel)
#     plt.grid(axis='y', alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(out_png, dpi=300)
#     plt.close()

# def shaded_mean_by_class(X_ul, y, fs, title, out_png):
#     """
#     X_ul: (n_trials, n_channels, n_samples) filtrado < UL_CUTOFF_HZ
#     Promedia sobre canales -> (n_trials, n_samples).
#     Dibuja mean±SEM por clase.
#     """
#     classes = np.unique(y)
#     trial_mean = X_ul.mean(axis=1)  # (n_trials, n_samples)
#     t = np.arange(trial_mean.shape[1]) / fs

#     plt.figure(figsize=(12, 5))
#     for c in classes:
#         M = trial_mean[y == c]  # (n_c, n_s)
#         if M.shape[0] == 0: 
#             continue
#         mu  = M.mean(axis=0)
#         sem = M.std(axis=0, ddof=1) / np.sqrt(M.shape[0])
#         plt.plot(t, mu, label=f"Clase {int(c)}", alpha=0.9)
#         plt.fill_between(t, mu - 1.96*sem, mu + 1.96*sem, alpha=0.15)
#     plt.xlabel("Tiempo (s)")
#     plt.ylabel("Amplitud (µV) — promedio sobre canales")
#     plt.title(title)
#     plt.grid(alpha=0.3)
#     plt.legend(ncol=min(len(classes), 5), fontsize=8)
#     plt.tight_layout()
#     plt.savefig(out_png, dpi=300)
#     plt.close()

# def scatter_time_feature(vec, y, title, ylabel, out_png):
#     """Dispersión de una feature (offset/slope/power) vs índice temporal de trial, coloreado por clase."""
#     idx = np.arange(len(vec))
#     plt.figure(figsize=(12, 4))
#     sc = plt.scatter(idx, vec, c=y, cmap='tab10', s=18, alpha=0.9)
#     plt.xlabel("Índice de trial (orden de carga)")
#     plt.ylabel(ylabel)
#     plt.title(title)
#     cb = plt.colorbar(sc); cb.set_label("Clase")
#     plt.grid(alpha=0.25)
#     plt.tight_layout()
#     plt.savefig(out_png, dpi=300)
#     plt.close()

# def anova_or_kruskal(data_per_class):
#     """Usa ANOVA si normalidad no es crítica; si hay tamaños muy dispares usa Kruskal (aquí ofrecido como alternativa)."""
#     # ANOVA 1 factor:
#     try:
#         stat_a, p_a = f_oneway(*data_per_class)
#     except Exception:
#         stat_a, p_a = np.nan, np.nan
#     # Kruskal (no paramétrico)
#     try:
#         stat_k, p_k = kruskal(*data_per_class)
#     except Exception:
#         stat_k, p_k = np.nan, np.nan
#     return (stat_a, p_a), (stat_k, p_k)

# # ----------------------- Carga -----------------------
# X_tr, y_tr, X_val, y_val, X_te, y_te = load_data_BCI(RAW_FILE, subject_id=SUBJECT_ID)
# print(f"Sujeto {SUBJECT_ID}  —  train {X_tr.shape}, val {X_val.shape}, test {X_te.shape}")

# # ----------------------- Filtrado UL -----------------------
# X_tr_ul = butter_lowpass(X_tr.astype(np.float32), UL_CUTOFF_HZ, FS, order=4)
# X_val_ul = butter_lowpass(X_val.astype(np.float32), UL_CUTOFF_HZ, FS, order=4)
# X_te_ul  = butter_lowpass(X_te.astype(np.float32), UL_CUTOFF_HZ, FS, order=4)

# # ----------------------- Features (TRAIN) -----------------------
# off_ch_tr, sl_ch_tr, off_g_tr, sl_g_tr = compute_offset_slope(X_tr, FS)
# pow_ch_tr, pow_g_tr = ul_power(X_tr_ul)

# # ----------------------- Stats globales por clase -----------------------
# off_tr_cls, classes = per_class_lists(off_g_tr, y_tr)
# sl_tr_cls,  _       = per_class_lists(sl_g_tr,  y_tr)
# pow_tr_cls, _       = per_class_lists(pow_g_tr, y_tr)

# (anova_off, kruskal_off) = anova_or_kruskal(off_tr_cls)
# (anova_sl,  kruskal_sl)  = anova_or_kruskal(sl_tr_cls)
# (anova_pw,  kruskal_pw)  = anova_or_kruskal(pow_tr_cls)

# print("\n=== TESTS globales (TRAIN) ===")
# print(f"Offset global — ANOVA:   F={anova_off[0]:.3f}, p={anova_off[1]:.3e}   | Kruskal: H={kruskal_off[0]:.3f}, p={kruskal_off[1]:.3e}")
# print(f"Slope global  — ANOVA:   F={anova_sl[0]:.3f},  p={anova_sl[1]:.3e}   | Kruskal: H={kruskal_sl[0]:.3f},  p={kruskal_sl[1]:.3e}")
# print(f"UL power      — ANOVA:   F={anova_pw[0]:.3f},  p={anova_pw[1]:.3e}   | Kruskal: H={kruskal_pw[0]:.3f},  p={kruskal_pw[1]:.3e}")

# # ----------------------- CSV por canal (TRAIN) -----------------------
# # ANOVA por canal para offset, slope y UL power
# csv_path = OUT_DIR / f"per_channel_stats_train_sub{SUBJECT_ID}.csv"
# with open(csv_path, "w", newline="") as f:
#     w = csv.writer(f)
#     w.writerow(["feature", "channel", "anova_F", "anova_p", "kruskal_H", "kruskal_p"])
#     for ch in range(X_tr.shape[1]):
#         # offset
#         off_cls_ch = [off_ch_tr[y_tr == c, ch] for c in classes]
#         (Fa, pa) = f_oneway(*off_cls_ch)
#         (Hk, pk) = kruskal(*off_cls_ch)
#         w.writerow(["offset", ch, float(Fa), float(pa), float(Hk), float(pk)])
#         # slope
#         sl_cls_ch  = [sl_ch_tr[y_tr == c, ch] for c in classes]
#         (Fa, pa)   = f_oneway(*sl_cls_ch)
#         (Hk, pk)   = kruskal(*sl_cls_ch)
#         w.writerow(["slope", ch, float(Fa), float(pa), float(Hk), float(pk)])
#         # power UL
#         pw_cls_ch  = [pow_ch_tr[y_tr == c, ch] for c in classes]
#         (Fa, pa)   = f_oneway(*pw_cls_ch)
#         (Hk, pk)   = kruskal(*pw_cls_ch)
#         w.writerow(["ul_power", ch, float(Fa), float(pa), float(Hk), float(pk)])
# print(f"CSV por canal guardado en: {csv_path}")

# # ----------------------- Figuras (TRAIN) -----------------------
# # Curvas mean±IC por clase de la señal UL (promedio sobre canales)
# shaded_mean_by_class(
#     X_tr_ul, y_tr, FS,
#     title=f"Sujeto {SUBJECT_ID} — Señal < {UL_CUTOFF_HZ} Hz (promedio canales) — media ± IC por clase (TRAIN)",
#     out_png=OUT_DIR / f"ul_meanIC_by_class_train_sub{SUBJECT_ID}.png"
# )

# # Evolución temporal de cada feature
# scatter_time_feature(off_g_tr, y_tr,
#     title=f"Sujeto {SUBJECT_ID} — Offset global por trial (TRAIN)",
#     ylabel="Offset global (µV)",
#     out_png=OUT_DIR / f"temporal_offset_global_train_sub{SUBJECT_ID}.png")

# scatter_time_feature(sl_g_tr, y_tr,
#     title=f"Sujeto {SUBJECT_ID} — Slope global por trial (TRAIN)",
#     ylabel="Slope global (µV/s)",
#     out_png=OUT_DIR / f"temporal_slope_global_train_sub{SUBJECT_ID}.png")

# scatter_time_feature(pow_g_tr, y_tr,
#     title=f"Sujeto {SUBJECT_ID} — Potencia < {UL_CUTOFF_HZ} Hz por trial (TRAIN)",
#     ylabel="UL power (varianza)",
#     out_png=OUT_DIR / f"temporal_ulpower_train_sub{SUBJECT_ID}.png")

# # Boxplots por clase
# save_boxplot(off_tr_cls, classes,
#     title=f"Offset global por clase — TRAIN — Sujeto {SUBJECT_ID}",
#     ylabel="Offset global (µV)",
#     out_png=OUT_DIR / f"box_offset_global_train_sub{SUBJECT_ID}.png")

# save_boxplot(sl_tr_cls, classes,
#     title=f"Slope global por clase — TRAIN — Sujeto {SUBJECT_ID}",
#     ylabel="Slope global (µV/s)",
#     out_png=OUT_DIR / f"box_slope_global_train_sub{SUBJECT_ID}.png")

# save_boxplot(pow_tr_cls, classes,
#     title=f"Potencia < {UL_CUTOFF_HZ} Hz por clase — TRAIN — Sujeto {SUBJECT_ID}",
#     ylabel="UL power (varianza)",
#     out_png=OUT_DIR / f"box_ulpower_train_sub{SUBJECT_ID}.png")

# # ----------------------- Repite rápido en VAL/TEST (global) -----------------------
# def quick_split_figs(X, y, split_tag):
#     X_ul = butter_lowpass(X.astype(np.float32), UL_CUTOFF_HZ, FS, order=4)
#     off_ch, sl_ch, off_g, sl_g = compute_offset_slope(X, FS)
#     _, pow_g = ul_power(X_ul)

#     off_cls, classes = per_class_lists(off_g, y)
#     sl_cls, _        = per_class_lists(sl_g, y)
#     pw_cls, _        = per_class_lists(pow_g, y)

#     (anova_off, kruskal_off) = anova_or_kruskal(off_cls)
#     (anova_sl,  kruskal_sl)  = anova_or_kruskal(sl_cls)
#     (anova_pw,  kruskal_pw)  = anova_or_kruskal(pw_cls)

#     print(f"\n=== {split_tag} ===")
#     print(f"Offset — ANOVA F={anova_off[0]:.3f}, p={anova_off[1]:.3e} | Kruskal H={kruskal_off[0]:.3f}, p={kruskal_off[1]:.3e}")
#     print(f"Slope  — ANOVA F={anova_sl[0]:.3f},  p={anova_sl[1]:.3e} | Kruskal H={kruskal_sl[0]:.3f},  p={kruskal_sl[1]:.3e}")
#     print(f"UL Pwr — ANOVA F={anova_pw[0]:.3f},  p={anova_pw[1]:.3e} | Kruskal H={kruskal_pw[0]:.3f},  p={kruskal_pw[1]:.3e}")

#     # figuras
#     shaded_mean_by_class(
#         X_ul, y, FS,
#         title=f"Sujeto {SUBJECT_ID} — Señal < {UL_CUTOFF_HZ} Hz — media ± IC por clase ({split_tag})",
#         out_png=OUT_DIR / f"ul_meanIC_by_class_{split_tag.lower()}_sub{SUBJECT_ID}.png"
#     )
#     save_boxplot(off_cls, classes,
#         title=f"Offset global por clase — {split_tag} — Sujeto {SUBJECT_ID}",
#         ylabel="Offset global (µV)",
#         out_png=OUT_DIR / f"box_offset_global_{split_tag.lower()}_sub{SUBJECT_ID}.png")
#     save_boxplot(sl_cls, classes,
#         title=f"Slope global por clase — {split_tag} — Sujeto {SUBJECT_ID}",
#         ylabel="Slope global (µV/s)",
#         out_png=OUT_DIR / f"box_slope_global_{split_tag.lower()}_sub{SUBJECT_ID}.png")
#     save_boxplot(pw_cls, classes,
#         title=f"Potencia < {UL_CUTOFF_HZ} Hz por clase — {split_tag} — Sujeto {SUBJECT_ID}",
#         ylabel="UL power (varianza)",
#         out_png=OUT_DIR / f"box_ulpower_{split_tag.lower()}_sub{SUBJECT_ID}.png")

# quick_split_figs(X_val, y_val, "VAL")
# quick_split_figs(X_te,  y_te,  "TEST")

# print(f"\nHecho. Figuras y CSV en: {OUT_DIR.resolve()}")





# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------


# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# from utils import load_data_BCI  # Debe devolver (X_train, y_train, X_val, y_val, X_test, y_test)
# from scipy.stats import f_oneway, ttest_ind
# import csv

# # ------------------------------------------------------------
# # Parámetros
# # ------------------------------------------------------------
# file = 'LP005'  # Cambia a 'RAW', 'BP', etc. para el archivo procesado
# fs = 256        # Frecuencia de muestreo (ajusta si fuera distinto)
# subject_id = 1  # Sujeto a cargar

# # Rutas (ajústalas si cambian)
# raw_file = Path("/home/w314/w314139/PROJECT/silent-speech-decoding/data/processed/BCI2020/BCI_raw.npz")
# processed_file = Path(f"/home/w314/w314139/PROJECT/silent-speech-decoding/data/processed/BCI2020/BCI_{file}.npz")

# # Carpeta de salida
# out_dir = Path("./bci_ultra_low_diagnostics")
# out_dir.mkdir(parents=True, exist_ok=True)

# # ------------------------------------------------------------
# # Carga de datos
# # ------------------------------------------------------------
# X_train, y_train, X_val, y_val, X_test, y_test = load_data_BCI(raw_file, subject_id=subject_id)
# X_train_proc, y_trainf, X_valf, y_valf, X_testf, y_testf = load_data_BCI(processed_file, subject_id=subject_id)

# # Sanidad mínima
# assert X_train.shape[1:] == X_train_proc.shape[1:], "Los shapes (chan, samples) deben coincidir"
# assert X_train.shape[0] == y_train.shape[0], "X_train e y_train no concuerdan"
# assert y_train.shape == y_trainf.shape, "Etiquetas train deben coincidir en longitud para raw/proc"

# # ------------------------------------------------------------
# # 1) Tu comparación de espectro (primer trial) — se mantiene
# # ------------------------------------------------------------
# x_raw = X_train[0]       # (channels, samples)
# x_proc = X_train_proc[0] # idem

# fft_raw = np.fft.fft(x_raw, axis=-1)
# fft_proc = np.fft.fft(x_proc, axis=-1)

# mag_raw = np.abs(fft_raw)
# mag_proc = np.abs(fft_proc)
# avg_spectrum_raw = np.mean(mag_raw, axis=0)
# avg_spectrum_proc = np.mean(mag_proc, axis=0)

# freqs = np.fft.fftfreq(x_raw.shape[1], d=1/fs)
# mask = (freqs >= 0) & (freqs <= 10)
# freqs_plot = freqs[mask]
# avg_spectrum_raw_plot = avg_spectrum_raw[mask]
# avg_spectrum_proc_plot = avg_spectrum_proc[mask]

# plt.figure(figsize=(10, 5))
# plt.plot(freqs_plot, avg_spectrum_raw_plot, label='Raw', alpha=0.8)
# plt.plot(freqs_plot, avg_spectrum_proc_plot, label=f'Processed ({file})', alpha=0.8)
# plt.title('Espectro de Frecuencia - Primer Trial')
# plt.xlabel('Frecuencia (Hz)')
# plt.ylabel('Magnitud')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(out_dir / f"spectrum_comparison_{file}.png", dpi=300)
# plt.close()

# # ------------------------------------------------------------
# # 2) Tu comparación temporal (primer canal del primer trial)
# # ------------------------------------------------------------
# signal_raw = X_train[0, 0, :]
# signal_proc = X_train_proc[0, 0, :]
# t = np.arange(signal_raw.shape[0]) / fs

# plt.figure(figsize=(10, 5))
# plt.plot(t, signal_raw, label='Raw', alpha=0.7)
# plt.plot(t, signal_proc, label=f'Processed ({file})', alpha=0.7)
# plt.title('Señal del Primer Canal - Primer Trial')
# plt.xlabel('Tiempo (s)')
# plt.ylabel('Amplitud (µV)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(out_dir / f"signal_comparison_channel0_trial0_{file}.png", dpi=300)
# plt.close()

# # ------------------------------------------------------------
# # 3) Offsets ultra-lentos por trial (media por canal)
# #    Haremos diagnósticos sobre TRAIN (puedes repetir con VAL/TEST)
# # ------------------------------------------------------------
# def compute_offsets_per_trial(X):
#     """
#     X: (n_trials, n_channels, n_samples)
#     return: offsets (n_trials, n_channels) = media temporal por canal
#     """
#     return X.mean(axis=2)

# offsets_raw_tr = compute_offsets_per_trial(X_train)         # (n_trials, n_channels)
# offsets_proc_tr = compute_offsets_per_trial(X_train_proc)   # idem

# # También un offset "global" por trial (media sobre canales)
# global_offset_raw = offsets_raw_tr.mean(axis=1)     # (n_trials,)
# global_offset_proc = offsets_proc_tr.mean(axis=1)   # (n_trials,)

# # ------------------------------------------------------------
# # 4) Test estadístico por canal: ¿difieren los offsets entre clases?
# #    - Si hay 2 clases: t-test Welch
# #    - Si hay >2: ANOVA de un factor
# #    Guardamos CSV con p-values
# # ------------------------------------------------------------
# classes = np.unique(y_train)
# n_channels = offsets_raw_tr.shape[1]

# def per_channel_class_test(offsets, y):
#     """
#     offsets: (n_trials, n_channels)
#     y: (n_trials,)
#     returns: list of dicts per channel with test name, stat, pvalue
#     """
#     results = []
#     for ch in range(n_channels):
#         samples = [offsets[y == c, ch] for c in classes]
#         # Ignora clases vacías por seguridad
#         samples = [s for s in samples if len(s) > 1]
#         if len(samples) < 2:
#             results.append({"channel": ch, "test": "NA", "stat": np.nan, "p": np.nan})
#             continue

#         if len(samples) == 2:
#             # Welch t-test
#             stat, p = ttest_ind(samples[0], samples[1], equal_var=False)
#             test_name = "Welch t-test"
#         else:
#             # ANOVA de un factor
#             stat, p = f_oneway(*samples)
#             test_name = "One-way ANOVA"

#         results.append({"channel": ch, "test": test_name, "stat": float(stat), "p": float(p)})
#     return results

# stats_raw = per_channel_class_test(offsets_raw_tr, y_train)
# stats_proc = per_channel_class_test(offsets_proc_tr, y_train)

# # Guardar CSV
# with open(out_dir / "offset_stats_raw_train.csv", "w", newline="") as f:
#     w = csv.DictWriter(f, fieldnames=["channel", "test", "stat", "p"])
#     w.writeheader()
#     w.writerows(stats_raw)

# with open(out_dir / f"offset_stats_{file}_train.csv", "w", newline="") as f:
#     w = csv.DictWriter(f, fieldnames=["channel", "test", "stat", "p"])
#     w.writeheader()
#     w.writerows(stats_proc)

# # Informe rápido por consola (opc.)
# sig_raw = [r for r in stats_raw if (not np.isnan(r["p"])) and r["p"] < 0.05]
# sig_proc = [r for r in stats_proc if (not np.isnan(r["p"])) and r["p"] < 0.05]
# print(f"[RAW] Canales con diferencias significativas (p<0.05): {len(sig_raw)} / {n_channels}")
# print(f"[{file}] Canales con diferencias significativas (p<0.05): {len(sig_proc)} / {n_channels}")

# # ------------------------------------------------------------
# # 5) Evolución temporal del offset global por trial (TRAIN)
# #    Si hay bloques, deberían aparecer segmentos coloreados por clase
# # ------------------------------------------------------------
# trial_idx = np.arange(len(y_train))

# plt.figure(figsize=(12, 4))
# sc = plt.scatter(trial_idx, global_offset_raw, c=y_train, s=18, alpha=0.9, cmap='tab10')
# plt.title('Evolución temporal del offset global (RAW) — Train')
# plt.xlabel('Índice de trial (supuesto orden de carga)')
# plt.ylabel('Offset global (µV)')
# cb = plt.colorbar(sc)
# cb.set_label('Clase')
# plt.tight_layout()
# plt.savefig(out_dir / "temporal_offset_global_RAW_train.png", dpi=300)
# plt.close()

# plt.figure(figsize=(12, 4))
# sc = plt.scatter(trial_idx, global_offset_proc, c=y_train, s=18, alpha=0.9, cmap='tab10')
# plt.title(f'Evolución temporal del offset global ({file}) — Train')
# plt.xlabel('Índice de trial (supuesto orden de carga)')
# plt.ylabel('Offset global (µV)')
# cb = plt.colorbar(sc)
# cb.set_label('Clase')
# plt.tight_layout()
# plt.savefig(out_dir / f"temporal_offset_global_{file}_train.png", dpi=300)
# plt.close()

# # ------------------------------------------------------------
# # 6) Boxplots por clase del offset global (TRAIN)
# # ------------------------------------------------------------
# def boxplot_global_offset_per_class(global_offset, y, title, out_png):
#     data = [global_offset[y == c] for c in classes]
#     labels = [f"Cls {int(c)} (n={len(global_offset[y==c])})" for c in classes]
#     plt.figure(figsize=(10, 5))
#     plt.boxplot(data, labels=labels, showmeans=True)
#     plt.title(title)
#     plt.ylabel("Offset global (µV)")
#     plt.grid(axis='y', alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(out_png, dpi=300)
#     plt.close()

# boxplot_global_offset_per_class(global_offset_raw, y_train,
#                                 "Offset global por clase (RAW) — Train",
#                                 out_dir / "box_offset_global_RAW_train.png")

# boxplot_global_offset_per_class(global_offset_proc, y_train,
#                                 f"Offset global por clase ({file}) — Train",
#                                 out_dir / f"box_offset_global_{file}_train.png")

# # ------------------------------------------------------------
# # 7) Heatmap offset (canales × trials) ordenado por clase (TRAIN)
# # ------------------------------------------------------------
# def heatmap_offsets(offsets, y, title, out_png):
#     # Ordena trials por clase (para ver bloques de color)
#     order = np.argsort(y)
#     off_sorted = offsets[order]            # (trials, channels)
#     y_sorted = y[order]
#     plt.figure(figsize=(12, 6))
#     plt.imshow(off_sorted.T, aspect='auto', interpolation='nearest', cmap='RdBu_r')
#     plt.colorbar(label='Offset (µV)')
#     plt.title(title + " — trials ordenados por clase")
#     plt.xlabel("Trials (ordenados por clase)")
#     plt.ylabel("Canales")
#     # ticks opcionales: marcas de cambio de clase
#     change_idx = np.where(np.diff(y_sorted) != 0)[0]
#     for x in change_idx:
#         plt.axvline(x + 0.5, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
#     plt.tight_layout()
#     plt.savefig(out_png, dpi=300)
#     plt.close()

# heatmap_offsets(offsets_raw_tr, y_train,
#                 "Heatmap de offsets por canal (RAW) — Train",
#                 out_dir / "heatmap_offsets_RAW_train.png")

# heatmap_offsets(offsets_proc_tr, y_train,
#                 f"Heatmap de offsets por canal ({file}) — Train",
#                 out_dir / f"heatmap_offsets_{file}_train.png")

# # ------------------------------------------------------------
# # 8) (Opcional) Repite análisis rápido en VAL/TEST
# #     — útil para ver si el patrón se mantiene
# # ------------------------------------------------------------
# def quick_report_split(X, y, split_name, tag):
#     offs = compute_offsets_per_trial(X)
#     glob = offs.mean(axis=1)

#     # ANOVA / t-test por canal
#     n_channels = offs.shape[1]
#     classes = np.unique(y)
#     def test_one(offsets, y):
#         results = []
#         for ch in range(n_channels):
#             samples = [offsets[y == c, ch] for c in classes]
#             samples = [s for s in samples if len(s) > 1]
#             if len(samples) < 2:
#                 results.append(np.nan)
#                 continue
#             if len(samples) == 2:
#                 stat, p = ttest_ind(samples[0], samples[1], equal_var=False)
#             else:
#                 stat, p = f_oneway(*samples)
#             results.append(p)
#         return np.array(results)

#     pvals = test_one(offs, y)
#     sig = np.sum((~np.isnan(pvals)) & (pvals < 0.05))
#     print(f"[{tag}] {split_name}: canales con p<0.05 = {sig}/{n_channels}")

#     # Boxplot
#     data = [glob[y == c] for c in classes]
#     labels = [f"Cls {int(c)} (n={len(glob[y==c])})" for c in classes]
#     plt.figure(figsize=(10, 5))
#     plt.boxplot(data, labels=labels, showmeans=True)
#     plt.title(f"Offset global por clase — {tag} — {split_name}")
#     plt.ylabel("Offset global (µV)")
#     plt.grid(axis='y', alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(out_dir / f"box_offset_global_{tag}_{split_name}.png", dpi=300)
#     plt.close()

# # Ejemplos rápidos
# quick_report_split(X_val, y_val, "VAL", "RAW")
# quick_report_split(X_test, y_test, "TEST", "RAW")
# quick_report_split(X_valf, y_valf, "VAL", file)
# quick_report_split(X_testf, y_testf, "TEST", file)

# print(f"Listo. Salidas guardadas en: {out_dir.resolve()}")

