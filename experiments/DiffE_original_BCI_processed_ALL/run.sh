#!/bin/bash
##----------------------- Descripción del trabajo -----------------------
#SBATCH --job-name=DiffE_original_BCI_processed_ALL
#SBATCH --partition=standard-gpu      # Cola con GPU
#SBATCH --gres=gpu:a100:1             # 1 GPU A100 (cambia a v100 si lo prefieres)
#SBATCH --ntasks=1                    # 1 tarea (proceso) Slurm
#SBATCH --cpus-per-task=4             # 4 hilos de CPU para esa tarea
#SBATCH --mem=16G                     # 16 GiB RAM en el nodo
#SBATCH --time=06:00:00               # Límite de tiempo
#SBATCH --output=/media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding/experiments/DiffE_original_BCI_processed_ALL/output.txt
#SBATCH --error=/media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding/experiments/DiffE_original_BCI_processed_ALL/error.txt
##-------------------------------------

# Move to project root
cd /media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding

# Activate virtual environment
source /media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding/venv/bin/activate

# Ensure unbuffered Python output for real-time logs
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---- Training command ----
srun python -u /media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding/models/DiffE_mod/main.py --exp-dir /media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding/experiments/DiffE_original_BCI_processed_ALL --dataset_file /media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding/data/processed/BCI2020/BCI_processed.npz --dataset BCI --device cuda:0 --num_epochs 100 --batch_train 250  --batch_eval 32 --seed 42 --alpha 10  --num_classes 5 --channels 64 --n_T 100 --ddpm_dim 128 --encoder_dim 64 --fc_dim 64
