#!/bin/bash
set -euo pipefail

echo "Generating output files..."
PYTHONPATH=. python src/utils/prepare_output.py "$@"
