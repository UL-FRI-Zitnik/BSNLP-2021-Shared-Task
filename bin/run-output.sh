#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=64
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/BSNLP-output-%J.out
#SBATCH --error=logs/BSNLP-output-%J.err
#SBATCH --job-name="BSNLP-output"

CONTAINER_IMAGE_PATH="$PWD/containers/pytorch-image-new.sqfs"

echo "$SLURM_JOB_ID -> Generating the output files for the models..."

# the following command opens a bash terminal of an already existing container
# with the current directory (.) mounted
srun \
    --container-image "$CONTAINER_IMAGE_PATH" \
    --container-mounts "$PWD":/workspace,/shared/datasets/rsdo:/data \
    --container-entrypoint /workspace/bin/exec-output.sh --lang "sl" --year "test_2021" --run-path "./data/runs/run_2496_slo_all"
    # --container-entrypoint /workspace/bin/exec-output.sh --lang "all" --year "test_2021" --run-path "./data/runs/run_l1o_2551"

echo "$SLURM_JOB_ID -> Done."
