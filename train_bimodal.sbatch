#!/bin/bash
#SBATCH --job-name=train-bimodal
#SBATCH --time=3-00:20:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --partition=frida
#SBATCH --account=lpt-phd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -o bimodal-train.out

CONTAINER_IMAGE_PATH=/shared/home/timotej.knez/containers/pytorch.sqfs
cd "${HOME}/llm-graph-construction"

srun \
  -A lpt -p frida \
  --container-image /shared/home/timotej.knez/containers/pytorch.sqfs \
  --container-mounts "$PWD":/workspace \
  bash -c "python main.py --method train_bimodal"
