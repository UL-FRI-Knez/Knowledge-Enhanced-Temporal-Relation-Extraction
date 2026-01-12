#!/bin/bash
#SBATCH --job-name=kg-llm
#SBATCH --time=3-00:20:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=frida
#SBATCH --account=lpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1

CONTAINER_IMAGE_PATH=/shared/home/timotej.knez/containers/pytorch.sqfs
cd "${HOME}/llm-graph-construction"

srun \
  -A lpt -p frida \
  --container-image /shared/home/timotej.knez/containers/pytorch.sqfs \
  --container-mounts "$PWD":/workspace \
  bash ./start_with_ollama.sh
