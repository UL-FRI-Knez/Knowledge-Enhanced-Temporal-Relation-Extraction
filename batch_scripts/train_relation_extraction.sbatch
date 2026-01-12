#!/bin/bash
#SBATCH --job-name=rel_det_rec
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --partition=frida
#SBATCH --account=lpt-phd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -o outputs/relation_detection.out

CONTAINER_IMAGE_PATH=/shared/home/timotej.knez/containers/gold.sqfs
cd "${HOME}/llm-graph-construction"

srun \
  -A lpt -p frida \
  --container-image /shared/home/timotej.knez/containers/gold.sqfs \
  --container-mounts "$PWD":/workspace \
  bash -c "pip uninstall -y evaluate;pip install evaluate psutil accelerate==0.10.0 wandb==0.15.12;python -m spacy download en_core_web_sm;bash start_with_ollama.sh --method train_relation_detection_and_prediction"
