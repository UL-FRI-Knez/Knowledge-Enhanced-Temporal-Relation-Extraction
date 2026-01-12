#!/bin/bash
#SBATCH --job-name=graph-analysis
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --partition=frida
#SBATCH --account=lpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -o analysis.out

CONTAINER_IMAGE_PATH=/shared/home/timotej.knez/containers/pytorch.sqfs
cd "${HOME}/llm-graph-construction"

srun \
  -A lpt -p frida \
  --container-image /shared/home/timotej.knez/containers/pytorch.sqfs \
  --container-mounts "$PWD":/workspace \
  python main.py --method precompute_graphs_for_analysis
