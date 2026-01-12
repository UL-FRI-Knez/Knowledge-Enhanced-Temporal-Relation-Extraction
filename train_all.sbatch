#!/bin/bash
#SBATCH --job-name=tre-eval
#SBATCH --time=6-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --partition=frida
#SBATCH --account=lpt-phd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -o all-train2.out

CONTAINER_IMAGE_PATH=/shared/home/timotej.knez/containers/pytorch.sqfs
cd "${HOME}/llm-graph-construction"

srun \
  -A lpt -p frida \
  --container-image /shared/home/timotej.knez/containers/pytorch.sqfs \
  --container-mounts "$PWD":/workspace \
  bash -c "pip uninstall -y transformers; pip uninstall -y accelerate; pip install transformers==4.20.1 accelerate==0.11.0 cryptpandas nltk lightning==2.0.8; ./start_with_ollama.sh --method train_all"
# glm: transformers==4.27.3
# ostalo: transformers==4.20.1