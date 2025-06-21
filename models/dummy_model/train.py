# silent-speech-decoding/models/dummy_model/train.py

import time
import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=str, required=True, help="Where to save outputs")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Simula entrenamiento
    print("Starting dummy training...")
    for epoch in range(60*3):
        print(f"Epoch {epoch+1}/3: training...")
        time.sleep(1)

    # Simula checkpoint
    weights_path = exp_dir / "model_weights.txt"
    with open(weights_path, "w") as f:
        f.write("fake-weights-123")

    # Simula métrica
    metrics_path = exp_dir / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("accuracy: 0.95\nloss: 0.05")

    print(f"Training completed. Weights saved to: {weights_path}")
    print(f"Metrics saved to: {metrics_path}")

if __name__ == "__main__":
    main()

