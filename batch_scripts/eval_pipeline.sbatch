#!/bin/bash
#SBATCH --job-name=eval-trp
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --partition=frida
#SBATCH --account=lpt-phd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -o eval.out

cd "${HOME}/llm-graph-construction"

srun \
  -A lpt -p frida \
  --container-image /shared/home/timotej.knez/containers/gold.sqfs \
  --container-mounts "$PWD":/workspace \
  bash -c "pip uninstall -y evaluate; pip install evaluate wandb; python -m spacy download en_core_web_sm; bash start_with_ollama.sh --method eval_pipeline"
