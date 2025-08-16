import os
import numpy as np

def calc_acc_and_len(filename):
    data = np.loadtxt(filename, delimiter=',')
    preds = data[:, 0].astype(int)
    labels = data[:, 1].astype(int)
    acc = np.mean(preds == labels)
    return acc, len(data)

resultados = []

for i in range(15):
    fname = f"{i}-Test.csv"
    if os.path.isfile(fname):
        acc, n = calc_acc_and_len(fname)
        print(f"{fname}: acc = {acc:.4f}, muestras = {n}")
        resultados.append((fname, acc, n))
    else:
        print(f"{fname} no encontrado.")

# Resumen general
print("\nResumen:")
for fname, acc, n in resultados:
    print(f"{fname}: acc = {acc:.4f}, muestras = {n}")
