#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=10
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/BSNLP-cluster-%J.out
#SBATCH --error=logs/BSNLP-cluster-%J.err
#SBATCH --job-name="BSNLP-cluster"

CONTAINER_IMAGE_PATH="$PWD/containers/container.sif"

# singularity run --nv $CONTAINER_IMAGE_PATH bin/exec-bert.sh --full-finetuning --epochs 5 --test # --train
singularity run --nv $CONTAINER_IMAGE_PATH bin/exec-clustering.sh
# singularity run --nv $CONTAINER_IMAGE_PATH bin/exec-pred.sh
