#!/bin/bash
set -euo pipefail

echo "Obtaining file structure ..."
PYTHONPATH=. python src/analyze/main.py "$@"
