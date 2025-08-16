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


# -------------------------------------------------------------------------------------------------------------------------


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

