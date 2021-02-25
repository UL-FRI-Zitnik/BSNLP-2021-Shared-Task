#!/bin/bash
set -euo pipefail

echo "Generating output files..."
PYTHONPATH=. python src/analyze/main.py "$@"
