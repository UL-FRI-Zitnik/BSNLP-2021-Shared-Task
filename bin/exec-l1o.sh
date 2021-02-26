#!/bin/bash
set -euo pipefail

echo "Starting the Leave-One-Out (L1O) training process..."
PYTHONPATH=. python src/train/trainer.py "$@"
