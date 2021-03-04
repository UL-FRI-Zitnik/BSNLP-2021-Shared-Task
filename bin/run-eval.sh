#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=5GB
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/BSNLP-eval-%J.out
#SBATCH --error=logs/BSNLP-eval-%J.err
#SBATCH --job-name="BSNLP-eval"

CONTAINER_IMAGE_PATH="$PWD/containers/pytorch-image-new.sqfs"

echo "$SLURM_JOB_ID -> Generating the clusters for the model..."

BUNDLE="multi_all"
DIR_PREFIX="data/evals/$SLURM_JOB_ID-$BUNDLE"
mkdir -p "$DIR_PREFIX/reports"
mkdir -p "$DIR_PREFIX/error-logs"
mkdir -p "$DIR_PREFIX/summaries"

# the following command opens a bash terminal of an already existing container
# with the current directory (.) mounted
srun \
    --container-image "$CONTAINER_IMAGE_PATH" \
    --container-mounts "$PWD":/workspace,/shared/datasets/rsdo:/data \
    --container-entrypoint /workspace/bin/exec-eval.sh "java-eval/data-$BUNDLE" "$DIR_PREFIX/reports" "$DIR_PREFIX/error-logs" "$DIR_PREFIX/summaries"

echo "$SLURM_JOB_ID -> Done."
