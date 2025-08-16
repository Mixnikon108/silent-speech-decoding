#!/bin/bash
##----------------------- Descripción del trabajo -----------------------
#SBATCH --job-name=otherModels_TOL_processed_ALL
#SBATCH --partition=standard-gpu      # Cola con GPU
#SBATCH --gres=gpu:a100:1             # 1 GPU A100 (cambia a v100 si lo prefieres)
#SBATCH --ntasks=1                    # 1 tarea (proceso) Slurm
#SBATCH --cpus-per-task=4             # 4 hilos de CPU para esa tarea
#SBATCH --mem=16G                     # 16 GiB RAM en el nodo
#SBATCH --time=06:00:00               # Límite de tiempo
#SBATCH --output=/media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding/experiments/otherModels_TOL_processed_ALL/output.txt
#SBATCH --error=/media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding/experiments/otherModels_TOL_processed_ALL/error.txt
##-------------------------------------

# Move to project root
cd /media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding

# Activate virtual environment
source /media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding/venv/bin/activate

# Ensure unbuffered Python output for real-time logs
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---- Training command ----
srun python -u /media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding/models/otherModels/main.py --n_time 512 --dataset_file /media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding/data/processed/TOL/TOL_processed.npz --dataset TOL --device cuda:0 --epochs 100 --n_channels 128 --n_classes 4 --out_csv /media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding/experiments/otherModels_TOL_processed_ALL/benchmark_results.csv