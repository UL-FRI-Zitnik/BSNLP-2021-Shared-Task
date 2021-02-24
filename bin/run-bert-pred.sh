#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16GB
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/NER-BERT-pred-%J.out
#SBATCH --error=logs/NER-BERT-pred-%J.err
#SBATCH --job-name="NER-BERT-pred"

set -euo pipefail

CONTAINER_IMAGE_PATH="$PWD/containers/pytorch-image.sqfs"

echo "$SLURM_JOB_ID -> Predicting from the model..."

# the following command opens a bash terminal of an already existing container
# with the current directory (.) mounted
srun \
    --container-image "$CONTAINER_IMAGE_PATH" \
    --container-mounts "$PWD":/workspace,/shared/datasets/rsdo:/data \
    --container-entrypoint /workspace/bin/exec-pred.sh --lang all --run-path "./data/runs/run_2021-02-16T11:43:57_bert-multi-cased_all-langs"
    # --container-entrypoint /workspace/bin/exec-pred.sh --lang sl --run-path "./data/runs/run_2021-02-19T08:02:08_slo-misc-models"
    # --container-entrypoint /workspace/bin/exec-pred.sh --lang sl --run-path "./data/runs/run_2021-02-17T11:42:19_slo-models"

echo "$SLURM_JOB_ID -> Done."

#wait
