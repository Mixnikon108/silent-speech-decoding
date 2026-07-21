#!/bin/bash
#SBATCH --job-name=TOL_preprocessing
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Move to project root
cd "$PROJECT_ROOT"

# Activate virtual environment
source /path/to/venv/bin/activate

cd "$PROJECT_ROOT/scripts/create_TOL_raw/"

# Ensure unbuffered Python output for real-time logs
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT/scripts:$PYTHONPATH"

# ---- Training command ----
python -u "$PROJECT_ROOT/scripts/create_TOL_raw/preprocessTOLraw.py"