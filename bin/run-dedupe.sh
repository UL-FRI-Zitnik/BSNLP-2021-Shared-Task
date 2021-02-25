#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=64
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/BSNLP-cluster-%J.out
#SBATCH --error=logs/BSNLP-cluster-%J.err
#SBATCH --job-name="BSNLP-cluster"

CONTAINER_IMAGE_PATH="$PWD/containers/pytorch-image-new.sqfs"

echo "$SLURM_JOB_ID -> Generating the clusters for the model..."

# the following command opens a bash terminal of an already existing container
# with the current directory (.) mounted
srun \
    --container-image "$CONTAINER_IMAGE_PATH" \
    --container-mounts "$PWD":/workspace,/shared/datasets/rsdo:/data \
    --container-entrypoint /workspace/bin/exec-clustering.sh --test --tsh 0.35 --run-path "data/deduper/runs/run_2021-02-24T09:05:35"

echo "$SLURM_JOB_ID -> Done."
