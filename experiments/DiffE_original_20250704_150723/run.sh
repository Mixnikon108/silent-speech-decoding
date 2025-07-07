#!/bin/bash
#SBATCH --job-name=DiffE_original
#SBATCH --output=/media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding/experiments/DiffE_original_20250704_150723/output.txt
#SBATCH --error=/media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding/experiments/DiffE_original_20250704_150723/error.txt
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Move to project root
cd /media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding

# Activate virtual environment
source /media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding/venv/bin/activate

# Ensure unbuffered Python output for real-time logs
export PYTHONUNBUFFERED=1

# ---- Training command ----
python -u /media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding/models/DiffE_original/main.py --exp-dir /media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding/experiments/DiffE_original_20250704_150723 --dataset_file /media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding/data/processed/BCI2020/filtered_BCI2020.npz --device cpu --num_epochs 800 --batch_train 250 --batch_eval 32 --seed 42 --alpha 10 --subject_id 1 --num_classes 5 --channels 64 --n_T 100 --ddpm_dim 128 --encoder_dim 64 --fc_dim 64
