#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64GB
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
    --container-entrypoint /workspace/bin/exec-clustering.sh

echo "$SLURM_JOB_ID -> Done."
