#!/bin/bash
set -euo pipefail

echo "Merging and annotating the files ..."
PYTHONPATH=. python src/transform/annotate_docs.py "$@"
