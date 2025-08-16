#!/bin/bash
#SBATCH --job-name=TOL_preprocessing
#SBATCH --output=/home/w314/w314139/PROJECT/silent-speech-decoding/scripts/create_TOL_raw/output.txt
#SBATCH --error=/home/w314/w314139/PROJECT/silent-speech-decoding/scripts/create_TOL_raw/error.txt
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Move to project root
cd /media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding

# Activate virtual environment
source /media/beegfs/home/w314/w314139/PROJECT/silent-speech-decoding/venv/bin/activate

cd /home/w314/w314139/PROJECT/silent-speech-decoding/scripts/create_TOL_raw/

# Ensure unbuffered Python output for real-time logs
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/w314/w314139/PROJECT/silent-speech-decoding/scripts:$PYTHONPATH

# ---- Training command ----
python -u /home/w314/w314139/PROJECT/silent-speech-decoding/scripts/create_TOL_raw/preprocessTOLraw.py