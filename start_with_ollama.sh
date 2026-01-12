#!/bin/bash

#%% md
apt install -y pciutils
curl -fsSL https://ollama.com/install.sh | sh

ollama serve &
ollama pull openchat:7b
python main.py "$@"
systemctl stop ollama.service