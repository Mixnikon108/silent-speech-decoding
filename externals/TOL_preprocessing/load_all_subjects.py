import sys, pathlib, numpy as np

# 1. Define rutas absolutas según tu estructura
DATA = pathlib.Path("/home/w314/w314139/PROJECT/silent-speech-decoding/data/raw/TOL")

from Python_Processing.Data_extractions import extract_data_from_subject
from Python_Processing.Data_processing  import select_time_window, transform_for_classificator

datatype = "EEG"
fs = 256
t_start = 1.5
t_end = 3.5
subjects = range(1, 11)  # sujetos 1 a 10

all_X, all_Y, all_subj = [], [], []

for N_S in subjects:
    print(f"\nCargando sujeto {N_S}...")
    try:
        X, Y = extract_data_from_subject(str(DATA), N_S, datatype)
        X = select_time_window(X, t_start=t_start, t_end=t_end, fs=fs)
        # Todas las clases
        Classes = [["Up"], ["Down"], ["Right"], ["Left"]]
        Conditions = [["Inner"], ["Inner"], ["Inner"], ["Inner"]]
        Xc, Yc = transform_for_classificator(X, Y, Classes, Conditions)
        all_X.append(Xc)
        all_Y.append(Yc)
        all_subj.append(np.ones_like(Yc) * N_S)
        print(f"   {Xc.shape[0]} trials cargados")
    except Exception as e:
        print(f"   Sujeto {N_S} no procesado: {e}")

# Unifica todos los sujetos
if all_X:
    X_all = np.concatenate(all_X, axis=0)
    Y_all = np.concatenate(all_Y, axis=0)
    Subjects_all = np.concatenate(all_subj, axis=0)

    print("\n==== RESUMEN ====")
    print(f"Trials totales: {X_all.shape[0]}")
    print(f"Shape datos:    {X_all.shape} (trials x canales x muestras)")
    print(f"Shape etiquetas:{Y_all.shape} (labels)")
    print(f"Subjects array: {Subjects_all.shape} (subject per trial)")

    # Opcional: guardar como npz
    np.savez("inner_speech_all_subjects.npz", X=X_all, Y=Y_all, Subjects=Subjects_all)
    print("✅ Guardado en inner_speech_all_subjects.npz")
else:
    print("No se han cargado sujetos correctamente.")
