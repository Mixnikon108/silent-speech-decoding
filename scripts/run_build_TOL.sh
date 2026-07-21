#!/bin/bash
# ────────────────────────────── Slurm ──────────────────────────────
#SBATCH --job-name=build_bcic2020
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00          # ajusta si hace falta
#SBATCH --cpus-per-task=4        # usa 4 hilos (carga/parsing .mat)
#SBATCH --mem=8G                 # memoria suficiente para 64 ch × 1 k × ~3 k trials

# ────────────────────────── Entorno Python ─────────────────────────
# Mueve a la raíz del proyecto
cd "$PROJECT_ROOT"

# Activa el entorno virtual
source /path/to/venv/bin/activate

# Salida sin buffer para ver logs en tiempo real
export PYTHONUNBUFFERED=1

# ─────────────────────────── Comando ───────────────────────────────
python -u "$PROJECT_ROOT/scripts/preprocess_TOL_mod.py"
