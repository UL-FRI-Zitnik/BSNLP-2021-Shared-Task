#!/bin/bash
set -euo pipefail

# Takes 2 args:
#   1. path to directory containing submission files
#   2. path to file where the report should be generated

java ./src/utils/ConsistencyCheck "$@"
