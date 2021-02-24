#!/bin/bash
set -euo pipefail

echo "Starting the BERT model prediction process..."
PYTHONPATH=. python src/eval/model_eval.py "$@"
