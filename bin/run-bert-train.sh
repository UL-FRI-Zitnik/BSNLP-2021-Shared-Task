#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/NER-BERT-train-%J.out
#SBATCH --error=logs/NER-BERT-train-%J.err
#SBATCH --job-name="NER-BERT-train"

set -euo pipefail

CONTAINER_IMAGE_PATH="$PWD/containers/pytorch-image-new.sqfs"

echo "$SLURM_JOB_ID -> Training the model..."

# the following command opens a bash terminal of an already existing container
# with the current directory (.) mounted
srun \
    --container-image "$CONTAINER_IMAGE_PATH" \
    --container-mounts "$PWD":/workspace,/shared/datasets/rsdo:/data \
    --container-entrypoint /workspace/bin/exec-bert.sh --epochs 5 --test --run-path "./data/runs/run_2021-02-17T11:42:19_slo-models"

echo "$SLURM_JOB_ID -> Done."

#wait
