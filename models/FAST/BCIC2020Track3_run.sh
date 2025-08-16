#!/bin/bash
##----------------------- Descripción del trabajo -----------------------
#SBATCH --job-name=DiffE_FAST
#SBATCH --partition=standard-gpu      # Cola con GPU
#SBATCH --gres=gpu:a100:1             # 1 GPU A100 (cambia a v100 si lo prefieres)
#SBATCH --ntasks=1                    # 1 tarea (proceso) Slurm
#SBATCH --cpus-per-task=4             # 4 hilos de CPU para esa tarea
#SBATCH --mem=16G                     # 16 GiB RAM en el nodo
#SBATCH --time=06:00:00               # Límite de tiempo
#SBATCH --output=/media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding/models/FAST/out-%j.log
#SBATCH --error=/media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding/models/FAST/err-%j.log
#SBATCH --mail-user=jorgechedo@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
##-----------------------------------------------------------------------

# 2) Activa tu entorno virtual/conda
source /media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding/venv/bin/activate

# 3) Ajustes de rendimiento de PyTorch/Lightning (opcionales)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1           # salida en tiempo real

# 4) Ve al directorio del proyecto
cd /media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding

# 5) Lanza el programa dentro de los recursos asignados
srun python -u models/FAST/BCIC2020Track3_train.py --gpu 0 --folds 0-15
