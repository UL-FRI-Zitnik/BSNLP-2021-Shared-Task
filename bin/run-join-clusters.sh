#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/BSNLP-join-clusters-%J.out
#SBATCH --error=logs/BSNLP-join-clusters-%J.err
#SBATCH --job-name="BSNLP-join-clusters"

CONTAINER_IMAGE_PATH="$PWD/containers/pytorch-image-new.sqfs"

echo "$SLURM_JOB_ID -> Generating the clusters for the model..."

# the following command opens a bash terminal of an already existing container
# with the current directory (.) mounted
srun \
    --container-image "$CONTAINER_IMAGE_PATH" \
    --container-mounts "$PWD":/workspace,/shared/datasets/rsdo:/data \
    --container-entrypoint /workspace/bin/exec-join-clusters.sh --year "test_2021" --lang "sl" --pred-path "./data/runs/run_2668_slo_misc-submission" --cluster-path "./data/deduper/runs/run_2508"
    # --container-entrypoint /workspace/bin/exec-join-clusters.sh --test --tsh 0.35 --train-chars --run-path "./data/deduper/runs/run_2605" --data-path "./data/runs/run_l1o_2551/predictions/bsnlp/bert-base-multilingual-cased-bsnlp-exclude-none-finetuned-5-epochs"
    # --container-entrypoint /workspace/bin/exec-clustering.sh --test --tsh 0.35 --train

echo "$SLURM_JOB_ID -> Done."
