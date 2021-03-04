#!/bin/bash
set -euo pipefail

echo "Starting the BSNLP clustering process..."
PYTHONPATH=. python src/utils/join_pred_cluster.py "$@"
