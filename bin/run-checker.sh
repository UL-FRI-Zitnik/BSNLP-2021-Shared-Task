#!/bin/bash
set -euo pipefail

# Takes 2 args:
#   1. path to directory containing submission files
#   2. path to file where the report should be generated
# Example usage [from repo root]:
#   ./bin/run-checker.sh data/challenge/2021/brexit/annotated/sl/ data/consistency_reports/report_2021-sl-brexit.txt

java src/utils/ConsistencyCheck "$@"
