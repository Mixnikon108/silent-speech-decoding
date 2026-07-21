#!/usr/bin/env python3
"""
run_experiments.py – Utility to dispatch a new experiment via SLURM
-------------------------------------------------------------------
Given the path to a model directory (e.g. silent-speech-decoding/models/modelA/),
this script will:

1. Create a timestamped experiment folder inside ``silent-speech-decoding/experiments``
   whose name is ``<model_name>_<YYYYMMDD_HHMMSS>``.
2. Inside that folder, write a fully‑parameterised ``run.sh`` ready for ``sbatch``.
   The script routes SLURM stdout/stderr to ``output.txt`` and ``error.txt`` in the
   **same** experiment folder so that every run is completely self‑contained.
3. Echo concise instructions so you can immediately launch the job:

       sbatch <experiment_dir>/run.sh

Anything your training code writes (checkpoints, logs, TensorBoard, …) should be
pointed to the same ``--exp-dir`` that ``run.sh`` passes, ensuring all artefacts
land inside the experiment folder.

Usage
-----
$ python scripts/run_experiments.py /absolute/path/to/model_dir \
        [--entry ENTRY_PY] [--time HH:MM:SS] [--gpus N] [--cpus N] \
        [--mem GB] [--partition PARTITION] [--venv REL_PATH]

Unrecognised flags are forwarded verbatim to the training script, so you can
add hyper‑parameters on the fly:

$ python scripts/run_experiments.py models/modelA --lr 1e-3 --epochs 30
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent  # silent-speech-decoding/
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse CLI arguments, returning (known, passthrough)."""
    parser = argparse.ArgumentParser(
        description="Create an experiment folder and SLURM script for a model run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model_path", type=Path, help="Path to the model directory")
    parser.add_argument(
        "--entry",
        type=str,
        default=None,
        help="Entry python file inside the model directory (auto‑detect train.py | main.py)",
    )
    parser.add_argument("--time", default="04:00:00", help="SBATCH --time value")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to request")
    parser.add_argument("--cpus", type=int, default=4, help="CPUs per task")
    parser.add_argument("--mem", type=int, default=16, help="Memory (GB)")
    parser.add_argument(
        "--partition", default=None, help="Optional SLURM partition (if cluster uses them)"
    )
    parser.add_argument(
        "--venv",
        default="venv",
        help="Relative path to the virtualenv to activate (from PROJECT_ROOT)",
    )

    # Split known vs unknown to forward remaining flags to training script
    return parser.parse_known_args()


def auto_detect_entry(model_dir: Path) -> str:
    """Return a reasonable training entry‑point inside *model_dir*."""
    for candidate in ("train.py", "main.py"):
        if (model_dir / candidate).is_file():
            return candidate
    raise FileNotFoundError(
        "Could not auto‑detect an entry script. Use --entry to specify it explicitly."
    )


def create_experiment_folder(model_dir: Path) -> Path:
    """Make timestamped folder inside experiments/ and return its Path."""
    model_name = model_dir.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = EXPERIMENTS_ROOT / f"{model_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=False)
    return exp_dir


def write_run_sh(
    *,
    exp_dir: Path,
    model_dir: Path,
    entry_file: str,
    slurm_time: str,
    gpus: int,
    cpus: int,
    mem: int,
    partition: str | None,
    venv: str,
    passthrough: list[str],
) -> Path:
    """Generate the run.sh SLURM script and make it executable."""
    run_sh = exp_dir / "run.sh"
    partition_line = f"#SBATCH --partition={partition}\n" if partition else ""

    run_script = textwrap.dedent(
        f"""#!/bin/bash
#SBATCH --job-name={model_dir.name}
#SBATCH --output={exp_dir}/output.txt
#SBATCH --error={exp_dir}/error.txt
#SBATCH --time={slurm_time}
{partition_line}#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}G

# Move to project root
cd {PROJECT_ROOT}

# Activate virtual environment
source {PROJECT_ROOT}/{venv}/bin/activate

# Ensure unbuffered Python output for real-time logs
export PYTHONUNBUFFERED=1

# ---- Training command ----
python -u {model_dir / entry_file} --exp-dir {exp_dir} {' '.join(passthrough)}
"""
    )

    run_sh.write_text(run_script)
    # chmod +x
    run_sh.chmod(run_sh.stat().st_mode | 0o111)
    return run_sh


def main():
    args, passthrough = parse_args()

    # Normalise and validate paths
    model_dir = args.model_path.expanduser().resolve()
    if not model_dir.is_dir():
        sys.exit(f"[✗] Model path does not exist or is not a directory: {model_dir}")

    # Ensure experiments root exists
    EXPERIMENTS_ROOT.mkdir(exist_ok=True)

    # Auto‑detect entry script if not provided
    entry_script = args.entry or auto_detect_entry(model_dir)

    # Create experiment directory
    exp_dir = create_experiment_folder(model_dir)

    # Write the SLURM shell script
    run_sh = write_run_sh(
        exp_dir=exp_dir,
        model_dir=model_dir,
        entry_file=entry_script,
        slurm_time=args.time,
        gpus=args.gpus,
        cpus=args.cpus,
        mem=args.mem,
        partition=args.partition,
        venv=args.venv,
        passthrough=passthrough,
    )

    print("\n[✓] Experiment folder ready:", exp_dir)
    print("[✓] SLURM script created:", run_sh)
    print("\nLaunch it with:\n   sbatch", run_sh, "\n")


if __name__ == "__main__":
    main()

