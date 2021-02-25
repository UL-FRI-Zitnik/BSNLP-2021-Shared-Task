#!/bin/bash
set -euo pipefail

echo "Creating dataset splits..."
PYTHONPATH=. python src/transform/create_splits.py "$@"
