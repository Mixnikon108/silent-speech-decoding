
# # import numpy as np
# # import matplotlib.pyplot as plt

# # filtered_file = "data/processed/BCI2020/4_filtered_BCI2020.npz"
# # raw_file      = "data/processed/BCI2020/raw_BCI2020.npz"

# # # Carga ambos archivos
# # data_filtered = np.load(filtered_file, allow_pickle=True)
# # data_raw      = np.load(raw_file, allow_pickle=True)




# # # Suponemos X_train tiene shape (n_trials, n_channels, n_samples)
# # data_fft = data_filtered["X_train"][:, :, :].T  # (n_samples,)

# # print("Dimensiones de la señal filtrada:", data_fft.shape)
# # N = data_fft.shape[2]        # nº muestras por epoch
# # fs = 256
# # freqs = np.fft.rfftfreq(N, 1/fs)

# # # FFT: magnitud media en todos los canales y trials
# # mag_fft = np.abs(np.fft.rfft(data_fft, axis=2)) / N       # (trials, chan, freqs)
# # mean_fft = mag_fft.mean(axis=(0, 1))                      # media sobre trials y canales

# # plt.figure(figsize=(10, 4))
# # plt.plot(freqs, mean_fft, label='Media FFT (trials+canales)', alpha=0.85)
# # plt.xlabel("Frecuencia (Hz)")
# # plt.ylabel("Magnitud FFT")
# # plt.title("Espectro medio tras preprocesado")
# # plt.xlim(0, fs/2)
# # plt.ylim(0, np.percentile(mean_fft, 95))  # Zoom automático sin el pico DC
# # plt.axvline(0.5, color='gray', ls='--', lw=0.8)
# # plt.axvline(125, color='gray', ls='--', lw=0.8)
# # plt.tight_layout()
# # plt.savefig("espectro_medio_preprocesado.png")
# # plt.close()
# # print("✅ Espectro medio guardado como 'espectro_medio_preprocesado.png'")




# # import numpy as np
# # import matplotlib.pyplot as plt

# # filtered_file = "data/processed/BCI2020/4_filtered_BCI2020.npz"

# # data_filtered = np.load(filtered_file, allow_pickle=True)
# # X = data_filtered["X_train"]    # shape: (n_samples, n_channels, n_trials)
# # print("Dimensiones de la señal filtrada:", X.shape)

# # fs = 256
# # N = X.shape[0]
# # freqs = np.fft.rfftfreq(N, 1/fs)

# # # FFT sobre samples
# # mag_fft = np.abs(np.fft.rfft(X, axis=0)) / N   # shape: (n_freqs, n_channels, n_trials)

# # # Media sobre trials para cada canal
# # mean_fft_channels = mag_fft.mean(axis=2)    # shape: (n_freqs, n_channels)

# # plt.figure(figsize=(12, 7))
# # for ch in range(mean_fft_channels.shape[1]):
# #     plt.plot(freqs, mean_fft_channels[:, ch], alpha=0.7, label=f'Canal {ch}' if ch < 10 else None)
# # plt.xlabel("Frecuencia (Hz)")
# # plt.ylabel("Magnitud FFT (media sobre trials)")
# # plt.title("Espectro de frecuencia por canal (media sobre todos los trials)")
# # plt.xlim(0, fs/2)
# # plt.ylim(0, np.percentile(mean_fft_channels, 99))  # Zoom automático
# # plt.axvline(0.5, color='gray', ls='--', lw=0.8)
# # plt.axvline(125, color='gray', ls='--', lw=0.8)
# # if mean_fft_channels.shape[1] <= 10:
# #     plt.legend()
# # plt.tight_layout()
# # plt.savefig("aa2.png")
# # plt.close()
# # print("✅ Figura guardada como 'espectro_todos_canales_media_trials.png'")



# # import numpy as np
# # import matplotlib.pyplot as plt

# # filtered_file = "data/processed/BCI2020/filtered_BCI2020.npz"
# # raw_file      = "data/processed/BCI2020/raw_BCI2020.npz"

# # # Carga ambos archivos
# # data_filtered = np.load(filtered_file, allow_pickle=True)
# # data_raw      = np.load(raw_file, allow_pickle=True)

# # Xf = data_filtered["X_train_f"]   # (n_samples, n_channels, n_trials)
# # Xr = data_raw["X_train"]

# # canal = 0
# # trial = 4000

# # signal_filt = Xf[:, canal, trial]
# # signal_raw  = Xr[:, canal, trial]

# # N = signal_raw.shape[0]
# # fs = 256
# # t = np.arange(N) / fs

# # # ----------- Señal temporal -----------
# # plt.figure(figsize=(12, 4))
# # plt.plot(t, signal_raw, label='Raw', alpha=0.8)
# # plt.plot(t, signal_filt, label='Filtrado', alpha=0.8)
# # plt.xlabel('Tiempo (s)')
# # plt.ylabel('Amplitud (µV)')
# # plt.title('Señal temporal — Canal 0, Trial 0')
# # plt.legend()
# # plt.tight_layout()
# # plt.savefig('canal0_trial0_signal.png')
# # plt.close()
# # print("✅ Señal temporal guardada como 'canal0_trial0_signal.png'")

# # # ----------- Espectro de frecuencia -----------
# # freqs = np.fft.rfftfreq(N, 1/fs)
# # fft_raw = np.abs(np.fft.rfft(signal_raw)) / N
# # fft_filt = np.abs(np.fft.rfft(signal_filt)) / N

# # plt.figure(figsize=(10, 4))
# # plt.plot(freqs, fft_raw, label='Raw', alpha=0.8)
# # plt.plot(freqs, fft_filt, label='Filtrado', alpha=0.8)
# # plt.xlabel('Frecuencia (Hz)')
# # plt.ylabel('Magnitud FFT')
# # plt.title('Espectro de frecuencias — Canal 0, Trial 0')
# # plt.xlim(0, fs/2)
# # plt.ylim(0, np.percentile(fft_filt, 98))  # Zoom automático sin el pico DC
# # plt.legend()
# # plt.tight_layout()
# # plt.savefig('canal0_trial0_fft.png')
# # plt.close()
# # print("✅ Espectro FFT guardado como 'canal0_trial0_fft.png'")





# # import numpy as np
# # import matplotlib.pyplot as plt

# # filtered_file = "data/processed/BCI2020/filtered_BCI2020.npz"
# # raw_file      = "data/processed/BCI2020/raw_BCI2020.npz"

# # # Carga ambos archivos
# # data_filtered = np.load(filtered_file, allow_pickle=True)
# # data_raw      = np.load(raw_file, allow_pickle=True)

# # Xf = data_filtered["X_train_f"]   # (n_samples, n_channels, n_trials)
# # Xr = data_raw["X_train"]

# # canal = 0

# # # --- Señal media sobre todos los trials para canal 0 ---
# # signal_filt = Xf[:, canal, :].mean(axis=1)  # (n_samples,)
# # signal_raw  = Xr[:, canal, :].mean(axis=1)

# # N = signal_raw.shape[0]
# # fs = 256
# # t = np.arange(N) / fs

# # # ----------- Señal temporal (media) -----------
# # plt.figure(figsize=(12, 4))
# # plt.plot(t, signal_raw, label='Raw', alpha=0.8)
# # plt.plot(t, signal_filt, label='Filtrado', alpha=0.8)
# # plt.xlabel('Tiempo (s)')
# # plt.ylabel('Amplitud media (µV)')
# # plt.title('Señal temporal MEDIA — Canal 0 (todos los trials)')
# # plt.legend()
# # plt.tight_layout()
# # plt.savefig('canal0_media_signal.png')
# # plt.close()
# # print("✅ Señal media temporal guardada como 'canal0_media_signal.png'")

# # # ----------- Espectro de frecuencia (media) -----------
# # freqs = np.fft.rfftfreq(N, 1/fs)
# # fft_raw = np.abs(np.fft.rfft(signal_raw)) / N
# # fft_filt = np.abs(np.fft.rfft(signal_filt)) / N

# # plt.figure(figsize=(10, 4))
# # plt.plot(freqs, fft_raw, label='Raw', alpha=0.8)
# # plt.plot(freqs, fft_filt, label='Filtrado', alpha=0.8)
# # plt.xlabel('Frecuencia (Hz)')
# # plt.ylabel('Magnitud FFT (media de trials)')
# # plt.title('Espectro de frecuencias MEDIA — Canal 0 (todos los trials)')
# # plt.xlim(0, fs/2)
# # plt.legend()
# # plt.tight_layout()
# # plt.savefig('canal0_media_fft.png')
# # plt.close()
# # print("✅ Espectro FFT medio guardado como 'canal0_media_fft.png'")



# # from pathlib import Path
# # import torch
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import torch.nn.functional as F

# # def pad_to_multiple(x: torch.Tensor, mult: int = 8):
# #     L = x.shape[-1]
# #     pad = (mult - L % mult) % mult
# #     if pad:
# #         x = F.pad(x, (0, pad), mode="reflect")
# #     return x

# # def zscore_norm(data: torch.Tensor) -> torch.Tensor:
# #     mean = torch.mean(data, dim=(1, 2), keepdim=True)
# #     std  = torch.std(data, dim=(1, 2), keepdim=True)
# #     return (data - mean) / (std + 1e-8)

# # def load_data(dataset_file: Path, subject_id: int = None):
# #     data = np.load(dataset_file)
# #     def _prep_X(arr):
# #         arr = np.transpose(arr, (2, 1, 0)).astype(np.float32)  # (trials, chan, samples)
# #         x = zscore_norm(torch.from_numpy(arr))
# #         x = pad_to_multiple(x, 8)
# #         return x
# #     def _prep_y(arr):
# #         return torch.from_numpy(np.argmax(arr, axis=0)).long()
# #     X_train = _prep_X(data["X_train"])
# #     y_train = _prep_y(data["y_train"])
# #     X_val   = _prep_X(data["X_val"])
# #     y_val   = _prep_y(data["y_val"])
# #     X_test  = _prep_X(data["X_test"])
# #     y_test  = _prep_y(data["y_test"])
# #     if subject_id is not None:
# #         i = subject_id - 1
# #         X_train = X_train[i*300:(i+1)*300]
# #         y_train = y_train[i*300:(i+1)*300]
# #         X_val   = X_val[i*50:(i+1)*50]
# #         y_val   = y_val[i*50:(i+1)*50]
# #         X_test  = X_test[(i+1)*50:(i+2)*50]
# #         y_test  = y_test[(i+1)*50:(i+2)*50]
# #     return X_train, y_train, X_val, y_val, X_test, y_test

# # # ---------------------- main ----------------------
# # filtered_file = Path("data/processed/BCI2020/4_filtered_BCI2020.npz")
# # raw_file      = Path("data/processed/BCI2020/raw_BCI2020.npz")

# # # Carga
# # X_train_filt, _, X_val_filt, _, X_test_filt, _ = load_data(filtered_file)
# # data_raw_np = np.load(raw_file, allow_pickle=True)["X_train"]  # (samples, chan, trials)

# # canal = 0
# # trial = 0

# # # Señal filtrada (z-score y padded) y cruda
# # signal_filt = X_train_filt[trial, canal].numpy()            # length = 800
# # signal_raw  = data_raw_np[:, canal, trial]                  # length = 795

# # # Recortamos la filtrada para igualar longitudes
# # signal_filt = signal_filt[: signal_raw.shape[0]]           # now length = 795

# # # Parámetros de muestreo
# # fs = 256
# # N  = signal_raw.shape[0]
# # t  = np.arange(N) / fs

# # # ----------- Señal temporal -----------
# # plt.figure(figsize=(12, 4))
# # plt.plot(t, signal_raw, label='Raw', alpha=0.8)
# # # plt.plot(t, signal_filt, label='Filtrado', alpha=0.8)
# # plt.xlabel('Tiempo (s)')
# # plt.ylabel('Amplitud (z-score)')
# # plt.title('Señal temporal — Canal 0, Trial 0')
# # plt.legend()
# # plt.tight_layout()
# # plt.savefig('1.png')
# # plt.close()
# # print("✅ Señal temporal guardada como '1.png'")

# # plt.figure(figsize=(12, 4))
# # # plt.plot(t, signal_raw, label='Raw', alpha=0.8)
# # plt.plot(t, signal_filt, label='Filtrado', alpha=0.8)
# # plt.xlabel('Tiempo (s)')
# # plt.ylabel('Amplitud (z-score)')
# # plt.title('Señal temporal — Canal 0, Trial 0')
# # plt.legend()
# # plt.tight_layout()
# # plt.savefig('2.png')
# # plt.close()
# # print("✅ Señal temporal guardada como '2.png'")

# # # ----------- Espectro de frecuencia -----------
# # freqs    = np.fft.rfftfreq(N, 1/fs)
# # fft_raw  = np.abs(np.fft.rfft(signal_raw)) / N
# # fft_filt = np.abs(np.fft.rfft(signal_filt)) / N

# # plt.figure(figsize=(10, 4))
# # plt.plot(freqs, fft_raw,  label='Raw',     alpha=0.8)
# # plt.plot(freqs, fft_filt, label='Filtrado',alpha=0.8)
# # plt.xlabel('Frecuencia (Hz)')
# # plt.ylabel('Magnitud FFT (z-score)')
# # plt.title('Espectro de frecuencias — Canal 0, Trial 0')
# # plt.xlim(0, fs/2)
# # plt.legend()
# # plt.tight_layout()
# # plt.savefig('canal0_trial0_fft_zscore.png')
# # plt.close()
# # print("✅ Espectro FFT guardado como 'canal0_trial0_fft_zscore.png'")





# # import numpy as np
# # import matplotlib.pyplot as plt
# # from pathlib import Path
# # from pathlib import Path
# # import torch
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import torch.nn.functional as F

# # def pad_to_multiple(x: torch.Tensor, mult: int = 8):
# #     L = x.shape[-1]
# #     pad = (mult - L % mult) % mult
# #     if pad:
# #         x = F.pad(x, (0, pad), mode="reflect")
# #     return x

# # def zscore_norm(data: torch.Tensor) -> torch.Tensor:
# #     mean = torch.mean(data, dim=(1, 2), keepdim=True)
# #     std  = torch.std(data, dim=(1, 2), keepdim=True)
# #     return (data - mean) / (std + 1e-8)

# # def load_data(dataset_file: Path, subject_id: int = None):
# #     data = np.load(dataset_file)
# #     def _prep_X(arr):
# #         arr = np.transpose(arr, (2, 1, 0)).astype(np.float32)  # (trials, chan, samples)
# #         x = zscore_norm(torch.from_numpy(arr))
# #         x = pad_to_multiple(x, 8)
# #         return x
# #     def _prep_y(arr):
# #         return torch.from_numpy(np.argmax(arr, axis=0)).long()
# #     X_train = _prep_X(data["X_train"])
# #     y_train = _prep_y(data["y_train"])
# #     X_val   = _prep_X(data["X_val"])
# #     y_val   = _prep_y(data["y_val"])
# #     X_test  = _prep_X(data["X_test"])
# #     y_test  = _prep_y(data["y_test"])
# #     if subject_id is not None:
# #         i = subject_id - 1
# #         X_train = X_train[i*300:(i+1)*300]
# #         y_train = y_train[i*300:(i+1)*300]
# #         X_val   = X_val[i*50:(i+1)*50]
# #         y_val   = y_val[i*50:(i+1)*50]
# #         X_test  = X_test[(i+1)*50:(i+2)*50]
# #         y_test  = y_test[(i+1)*50:(i+2)*50]
# #     return X_train, y_train, X_val, y_val, X_test, y_test



# # filtered_file = Path("data/processed/BCI2020/4_filtered_BCI2020.npz")
# # raw_file      = Path("data/processed/BCI2020/raw_BCI2020.npz")

# # # Cargar datos procesados y crudos
# # X_train_filt, _, _, _, _, _ = load_data(filtered_file)
# # X_train_raw, _, _, _, _, _ = load_data(raw_file)
# # X_train_raw = X_train_raw.numpy().T  # Convertir a numpy para compatibilidad

# # canal = 0
# # n_trials = min(X_train_filt.shape[0], X_train_raw.shape[2])  # por si hay padding


# # # RAW
# # max_raw = []
# # min_raw = []
# # for trial in range(n_trials):
# #     signal = X_train_raw[:, canal, trial]
# #     max_raw.append(np.max(signal))
# #     min_raw.append(np.min(signal))

# # # FILTRADO (recorta si hay padding)
# # max_filt = []
# # min_filt = []
# # for trial in range(n_trials):
# #     signal = X_train_filt[trial, canal].numpy()
# #     signal = signal[:X_train_raw.shape[0]]
# #     max_filt.append(np.max(signal))
# #     min_filt.append(np.min(signal))

# # # --------- PLOT ---------
# # fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=False)

# # # Raw
# # axs[0].scatter(range(n_trials), max_raw, color='red',  label='Máximo', s=18)
# # axs[0].scatter(range(n_trials), min_raw, color='green',label='Mínimo', s=18)
# # axs[0].set_title('Extremos por trial — Canal 0 (RAW)')
# # axs[0].set_xlabel('Trial')
# # # set ylim to just 99 percentile of max and min
# # # min_raw = np.percentile(min_raw, 1)
# # # max_raw = np.percentile(max_raw, 99)
# # # axs[0].set_ylim(min_raw, max_raw)

# # axs[0].set_ylabel('Valor extremo')
# # axs[0].legend()
# # axs[0].grid(True)

# # # Filtrado
# # axs[1].scatter(range(n_trials), max_filt, color='red',  label='Máximo', s=18)
# # axs[1].scatter(range(n_trials), min_filt, color='green',label='Mínimo', s=18)
# # axs[1].set_title('Extremos por trial — Canal 0 (FILTRADO)')
# # axs[1].set_xlabel('Trial')
# # axs[1].legend()
# # axs[1].grid(True)

# # plt.tight_layout()
# # plt.savefig('scatter_extremos_trial0.png')
# # plt.close()
# # print("✅ Scatterplot de extremos guardado como 'scatter_extremos_trial0.png'")




# # extremos_vs_clase.py
# from pathlib import Path
# import numpy as np
# import torch
# from scipy.stats import kruskal
# from tabulate import tabulate   # pip install tabulate si no lo tienes
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# from pathlib import Path
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import torch.nn.functional as F


# def pad_to_multiple(x: torch.Tensor, mult: int = 8):
#     L = x.shape[-1]
#     pad = (mult - L % mult) % mult
#     if pad:
#         x = F.pad(x, (0, pad), mode="reflect")
#     return x

# def zscore_norm(data: torch.Tensor) -> torch.Tensor:
#     mean = torch.mean(data, dim=(1, 2), keepdim=True)
#     std  = torch.std(data, dim=(1, 2), keepdim=True)
#     return (data - mean) / (std + 1e-8)

# def load_data(dataset_file: Path, subject_id: int = None):
#     data = np.load(dataset_file)
#     def _prep_X(arr):
#         arr = np.transpose(arr, (2, 1, 0)).astype(np.float32)  # (trials, chan, samples)
#         x = zscore_norm(torch.from_numpy(arr))
#         x = pad_to_multiple(x, 8)
#         return x
#     def _prep_y(arr):
#         return torch.from_numpy(np.argmax(arr, axis=0)).long()
#     X_train = _prep_X(data["X_train"])
#     y_train = _prep_y(data["y_train"])
#     X_val   = _prep_X(data["X_val"])
#     y_val   = _prep_y(data["y_val"])
#     X_test  = _prep_X(data["X_test"])
#     y_test  = _prep_y(data["y_test"])
#     if subject_id is not None:
#         i = subject_id - 1
#         X_train = X_train[i*300:(i+1)*300]
#         y_train = y_train[i*300:(i+1)*300]
#         X_val   = X_val[i*50:(i+1)*50]
#         y_val   = y_val[i*50:(i+1)*50]
#         X_test  = X_test[(i+1)*50:(i+2)*50]
#         y_test  = y_test[(i+1)*50:(i+2)*50]
#     return X_train, y_train, X_val, y_val, X_test, y_test


# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# # --- Carga sujeto 1 con z-score, como lo ve tu modelo ---
# raw_file = Path("data/processed/BCI2020/raw_BCI2020.npz")
# X_train, y_train, *_ = load_data(raw_file, subject_id=1)  # (n_trials, n_channels, n_samples)
# X_train = X_train.numpy()
# labels = y_train.numpy()

# canal = 0  # el canal que quieras (puedes cambiarlo)
# n_trials = X_train.shape[0]

# max_vals = X_train[:, canal, :].max(axis=1)  # (n_trials,)
# min_vals = X_train[:, canal, :].min(axis=1)

# # Agrupa por clase
# classes = np.unique(labels)
# max_groups = [max_vals[labels == c] for c in classes]
# min_groups = [min_vals[labels == c] for c in classes]

# # --------- BOXPLOT ----------
# fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
# axs[0].boxplot(max_groups, labels=[f"Clase {c}" for c in classes])
# axs[0].set_title(f"Máximos por clase — Canal {canal}")
# axs[0].set_ylabel("Máximo (z-score)")
# axs[1].boxplot(min_groups, labels=[f"Clase {c}" for c in classes])
# axs[1].set_title(f"Mínimos por clase — Canal {canal}")
# axs[1].set_ylabel("Mínimo (z-score)")
# plt.suptitle(f"Distribución de extremos por clase — Canal {canal}")
# plt.tight_layout()
# plt.savefig("boxplot_extremos_canal0.png")
# plt.close()
# print("✅ Boxplot de extremos guardado como 'boxplot_extremos_canal0.png'")


