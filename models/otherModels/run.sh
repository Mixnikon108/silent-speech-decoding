#!/bin/bash
##----------------------- Descripción del trabajo -----------------------
#SBATCH --job-name=ssd_baselines
#SBATCH --partition=standard-gpu      # Cola con GPU
#SBATCH --gres=gpu:a100:1             # 1 GPU A100 (cambia a v100 si lo prefieres)
#SBATCH --ntasks=1                    # 1 tarea (proceso) Slurm
#SBATCH --cpus-per-task=4             # 4 hilos de CPU para esa tarea
#SBATCH --mem=16G                     # 16 GiB RAM en el nodo
#SBATCH --time=06:00:00               # Límite de tiempo
#SBATCH --output=models/otherModels/output.txt
#SBATCH --error=models/otherModels/error.txt
##-------------------------------------

# Move to project root
PROJECT_ROOT=/path/to/project
cd "$PROJECT_ROOT"

# Activate virtual environment
source /path/to/venv/bin/activate

# Ensure unbuffered Python output for real-time logs
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---- Training command (baseline benchmark) ----
python -u "$PROJECT_ROOT"/models/otherModels/main.py \
  --dataset_file "$PROJECT_ROOT"/data/processed/BCI2020/BCI_raw.npz \
  --device cuda:0 --epochs 100

