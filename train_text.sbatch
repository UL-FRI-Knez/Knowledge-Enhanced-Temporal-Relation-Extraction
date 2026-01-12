#!/bin/bash
#SBATCH --job-name=train-text
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --partition=frida
#SBATCH --account=lpt-phd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -o text-train.out

CONTAINER_IMAGE_PATH=/shared/home/timotej.knez/containers/pytorch.sqfs
cd "${HOME}/llm-graph-construction"

srun \
  -A lpt -p frida \
  --container-image /shared/home/timotej.knez/containers/pytorch.sqfs \
  --container-mounts "$PWD":/workspace \
  bash -c "pip uninstall -y transformers; pip uninstall -y accelerate; pip install transformers; pip install accelerate; python main.py --method train_text"
