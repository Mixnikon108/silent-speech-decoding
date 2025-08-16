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

