#!/bin/bash
# ────────────────────────────── Slurm ──────────────────────────────
#SBATCH --job-name=build_bcic2020
#SBATCH --output=/media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding/scripts/output_TOL.txt
#SBATCH --error=/media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding/scripts/errors_TOL.txt
#SBATCH --time=02:00:00          # 10 min, ajusta si hace falta
#SBATCH --cpus-per-task=4        # usa 4 hilos (carga/parsing .mat)
#SBATCH --mem=8G                 # memoria suficiente para 64 ch × 1 k × ~3 k trials

# ────────────────────────── Entorno Python ─────────────────────────
# Mueve a la raíz del proyecto
cd /media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding

# Activa tu venv (ajusta nombre si fuera distinto)
source /media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding/venv/bin/activate

# Salida sin buffer para ver logs en tiempo real
export PYTHONUNBUFFERED=1

# ─────────────────────────── Comando ───────────────────────────────
python -u /media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding/scripts/preprocess_TOL.py
