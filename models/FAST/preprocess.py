# import os
# import numpy as np
# import h5py

# # Rutas
# SRC_FILE = '/home/w314/w314139/PROJECT/silent-speech-decoding/data/processed/BCI2020/filtered_BCI2020.npz'
# OUT_FOLDER = './Processed'
# OUT_FILE = f'{OUT_FOLDER}/BCI2020_filtered.h5'
# SUBJECTS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']

# # Cargar archivo
# data = np.load(SRC_FILE)

# X_train = data['X_train']    # (795, 64, 4500)
# y_train = data['y_train']    # (5, 4500)
# X_val = data['X_val']        # (795, 64, 750)
# y_val = data['y_val']        # (5, 750)

# print(f"Datos cargados: X_train {X_train.shape}, y_train {y_train.shape}, "
#       f"X_val {X_val.shape}, y_val {y_val.shape}")

# os.makedirs(OUT_FOLDER, exist_ok=True)

# with h5py.File(OUT_FILE, 'w') as f:
#     for k, subj in enumerate(SUBJECTS):
#         # Extraer y concatenar sólo train y val
#         X_tr = X_train[:, :, k*300:(k+1)*300]
#         y_tr = y_train[:, k*300:(k+1)*300]
#         X_va = X_val[:, :, k*50:(k+1)*50]
#         y_va = y_val[:, k*50:(k+1)*50]

#         X = np.concatenate([X_tr, X_va], axis=2)   # (795, 64, 350)
#         Y = np.concatenate([y_tr, y_va], axis=1)   # (5, 350)

#         # Transpose para trials, channels, samples: (350, 64, 795)
#         X = X.transpose(2, 1, 0)
#         # Pad al final del eje de samples (eje 2) para llegar a 800
#         if X.shape[2] < 800:
#             pad_width = ((0, 0), (0, 0), (0, 800 - X.shape[2]))
#             X = np.pad(X, pad_width, mode='constant')
#         # Y a 1D (de one-hot si necesario)
#         if Y.shape[0] <= 10:
#             Y = np.argmax(Y, axis=0)
#         else:
#             Y = Y.flatten()

#         f.create_dataset(f'{subj}/X', data=X)
#         f.create_dataset(f'{subj}/Y', data=Y.astype(np.uint8))
#         print(f"Sujeto {subj}: X shape {X.shape}, Y shape {Y.shape}, etiquetas: {np.unique(Y, return_counts=True)}")

# print(f"Archivo guardado en {OUT_FILE}")










# import h5py
# import numpy as np

# # Paths de los archivos a comparar
# file1 = './Processed/BCI2020.h5'    # Raw
# file2 = './Processed/BCI2020_filtered.h5'  # Procesado

# def compare_files(file1, file2):
#     with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
#         subjects1 = sorted(list(f1.keys()))
#         subjects2 = sorted(list(f2.keys()))
#         assert subjects1 == subjects2, "Diferentes sujetos encontrados en los archivos."
#         print(f"Comparando {len(subjects1)} sujetos...\n")

#         for subj in subjects1:
#             Y1 = f1[f"{subj}/Y"][:]
#             Y2 = f2[f"{subj}/Y"][:]

#             # Si Y tiene shape (5, 400): Pasar a 1D por columna mayoritaria
#             if len(Y1.shape) == 2:
#                 Y1 = np.argmax(Y1, axis=0) if Y1.shape[0] <= 10 else Y1.flatten()
#             if len(Y2.shape) == 2:
#                 Y2 = np.argmax(Y2, axis=0) if Y2.shape[0] <= 10 else Y2.flatten()

#             # Shape
#             ok_shape = Y1.shape == Y2.shape
#             # Dtype
#             ok_dtype = Y1.dtype == Y2.dtype
#             # Etiquetas y recuentos
#             uniq1, counts1 = np.unique(Y1, return_counts=True)
#             uniq2, counts2 = np.unique(Y2, return_counts=True)
#             ok_labels = np.array_equal(uniq1, uniq2) and np.array_equal(counts1, counts2)

#             print(f"Sujeto {subj}:")
#             print(f"    Shape Y: {Y1.shape} {'OK' if ok_shape else '❌'}")
#             print(f"    Dtype Y: {Y1.dtype} {'OK' if ok_dtype else '❌'}")
#             print(f"    Etiquetas: {uniq1} ({counts1}) {'OK' if ok_labels else '❌'}")
#             if not ok_labels:
#                 print(f"    >> Diferencia: {uniq1} vs {uniq2}, counts {counts1} vs {counts2}")
#             # Si quieres comparar shape de X:
#             X1 = f1[f"{subj}/X"]
#             X2 = f2[f"{subj}/X"]
#             print(f"    Shape X: {X1.shape} vs {X2.shape}")

#             print("")

# compare_files(file1, file2)




import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# ---------- paths ----------
file1 = './Processed/BCI2020.h5'          # raw
file2 = './Processed/BCI2020_filtered.h5'    # filtrado
subject = '01'                                     # sujeto que comparamos
Fs = 256.0                                         # Hz, ajusta si tu dataset usa otra Fs
# ---------------------------

def mean_spectrum(X, fs):
    """
    X: ndarray (trials, channels, samples)
    return: frecuencia (Hz), espectro medio (magnitud)
    """
    # Quitar la media de cada señal para evitar pico DC
    X = X - X.mean(axis=2, keepdims=True)

    # FFT a lo largo del eje temporal (último)
    spec = np.abs(rfft(X, axis=2))          # (trials, channels, N_freq)
    spec_mean = spec.mean(axis=(0, 1))      # media sobre trials y canales
    freqs = rfftfreq(X.shape[2], 1/fs)      # eje de frecuencias
    return freqs, spec_mean

with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
    # ----- etiquetas (por si quieres volver a verlas) -----
    y_raw  = f1[f"{subject}/Y"][:]
    y_filt = f2[f"{subject}/Y"][:]
    print("Etiquetas raw:",  np.unique(y_raw,  return_counts=True))
    print("Etiquetas filt:", np.unique(y_filt, return_counts=True))

    # ------------- espectros -------------
    X_raw  = f1[f"{subject}/X"][:]   # (trials, ch, samples)
    X_filt = f2[f"{subject}/X"][:]

    freqs, spec_raw  = mean_spectrum(X_raw,  Fs)
    _,     spec_filt = mean_spectrum(X_filt, Fs)

    # ---------- guardar gráficas ----------
    # 1) Histograma etiquetas
    plt.figure(figsize=(10,4))
    bins = np.arange(6)-0.5
    plt.subplot(1,2,1)
    plt.hist(y_raw , bins=bins, rwidth=0.8, color='steelblue')
    plt.title('Etiquetas original')
    plt.xlabel('Clase'); plt.ylabel('Nº muestras'); plt.xticks(range(5))

    plt.subplot(1,2,2)
    plt.hist(y_filt, bins=bins, rwidth=0.8, color='seagreen')
    plt.title('Etiquetas filtrado')
    plt.xlabel('Clase'); plt.ylabel('Nº muestras'); plt.xticks(range(5))

    plt.tight_layout()
    plt.savefig('sujeto1_etiquetas_comparacion.png')
    plt.close()

    # 2) Espectro medio
    plt.figure(figsize=(8,4))
    plt.plot(freqs, spec_raw , label='Original',  alpha=0.8)
    plt.plot(freqs, spec_filt, label='Filtrado',  alpha=0.8)
    plt.xlim(0, 100)          # muestra hasta 100 Hz; cambia si quieres
    plt.title('Espectro medio canales·trials – sujeto 1')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud')
    plt.legend()
    plt.tight_layout()
    plt.savefig('sujeto1_espectro_comparacion.png')
    plt.close()

print("Gráficos guardados como:")
print("  sujeto1_etiquetas_comparacion.png")
print("  sujeto1_espectro_comparacion.png")
