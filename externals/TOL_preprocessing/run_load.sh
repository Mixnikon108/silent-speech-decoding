#!/bin/bash
#SBATCH --job-name=load_subjects
#SBATCH --output=load_subjects_output.txt
#SBATCH --error=load_subjects_error.txt
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

# 1) activar venv
source /home/w314/w314139/PROJECT/silent-speech-decoding/venv/bin/activate

# 2) añadir **la carpeta externals** al PYTHONPATH
export PYTHONPATH="/home/w314/w314139/PROJECT/silent-speech-decoding/externals:$PYTHONPATH"

# 3) lanzar el script
python /home/w314/w314139/PROJECT/silent-speech-decoding/externals/TOL_preprocessing/load_all_subjects.py

deactivate
