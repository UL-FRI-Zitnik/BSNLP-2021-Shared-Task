# build a new container:
singularity build <local_target>.sif docker://<URL>
# e.g.
singularity build ./containers/container.sif docker://pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

# install dependencies within the container
singularity exec ./containers/container.sif pip install -r requirements.txt

# run the container
singularity run --nv ./containers/sing-container.sif

# submit a job to SLURM using a singularity container
sbatch -p compute --gres=gpu:1 --parsable ./bin/run-singularity-container.sh
sbatch -p compute -c 10  --parsable ./bin/run-singularity-container.sh
