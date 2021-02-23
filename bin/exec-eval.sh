#!/bin/bash
set -euo pipefail

java -cp java-eval/bsnlp-ner-evaluator-19.0.4.jar sigslav.BNEvaluator "$@"
