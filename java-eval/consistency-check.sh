#!/bin/bash
set -euo pipefail

java -cp bsnlp-ner-evaluator-19.0.4.jar sigslav.ConsistencyCheck data/golden/brexit/cs out
