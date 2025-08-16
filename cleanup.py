import os
from pathlib import Path
import subprocess

# Ruta al directorio de experiments
experiments_dir = Path("/home/w314/w314139/PROJECT/silent-speech-decoding/experiments")

# # Recorre cada subcarpeta dentro de experiments
# for subdir in experiments_dir.iterdir():
#     if subdir.is_dir():
#         print(f"Procesando: {subdir}")
#         for item in subdir.iterdir():
#             # Saltar si es run.sh
#             if item.name == "run.sh":
#                 continue
#             # Eliminar archivos
#             if item.is_file():
#                 item.unlink()
#             # Eliminar carpetas recursivamente
#             elif item.is_dir():
#                 for root, dirs, files in os.walk(item, topdown=False):
#                     for file in files:
#                         Path(root, file).unlink()
#                     for d in dirs:
#                         Path(root, d).rmdir()
#                 item.rmdir()



# Recorre cada subdirectorio
for subdir in experiments_dir.iterdir():
    if subdir.is_dir():
        run_sh = subdir / "run.sh"
        if run_sh.exists():
            print(f"Enviando trabajo: {run_sh}")
            try:
                subprocess.run(["sbatch", str(run_sh)], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error al enviar {run_sh}: {e}")
