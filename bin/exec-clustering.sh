#!/bin/bash
set -euo pipefail

echo "Starting the BSNLP clustering process..."
PYTHONPATH=. python src/matching/match_dedupe-all-chars.py "$@"
