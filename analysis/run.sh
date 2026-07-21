#!/bin/bash
##----------------------- Descripción del trabajo -----------------------
#SBATCH --job-name=ssd_analysis
#SBATCH --partition=standard-gpu      # Cola con GPU
#SBATCH --gres=gpu:a100:1             # 1 GPU A100 (cambia a v100 si lo prefieres)
#SBATCH --ntasks=1                    # 1 tarea (proceso) Slurm
#SBATCH --cpus-per-task=4             # 4 hilos de CPU para esa tarea
#SBATCH --mem=16G                     # 16 GiB RAM en el nodo
#SBATCH --time=06:00:00               # Límite de tiempo
#SBATCH --output=/path/to/silent-speech-decoding/analysis/output.txt
#SBATCH --error=/path/to/silent-speech-decoding/analysis/error.txt
##-------------------------------------

# Raíz del proyecto (ajusta a tu instalación)
PROJECT_ROOT=/path/to/silent-speech-decoding

# Move to project root
cd "$PROJECT_ROOT"

# Activate virtual environment
source "$PROJECT_ROOT/venv/bin/activate"

# Ensure unbuffered Python output for real-time logs
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---- Training command ----

srun python -u "$PROJECT_ROOT/analysis/cnn_test.py" \
  --dataset BCI \
  --dataset_file "$PROJECT_ROOT/data/processed/BCI2020/BCI_raw.npz" \
  --subject_id 1 \
  --device cuda:0 \
  --epochs 100 \
  --batch_train 64 \
  --batch_eval 64 \
  --n_time 800 \
  --n_classes 5