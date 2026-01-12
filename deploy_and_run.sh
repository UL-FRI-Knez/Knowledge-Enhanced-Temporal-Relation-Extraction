#!/bin/bash
scp main.py timotej.knez@frida:/shared/home/timotej.knez/llm-graph-construction
rsync -av -r graph_building/local_graph timotej.knez@frida:/shared/home/timotej.knez/llm-graph-construction/graph_building
rsync -av -r date2vec timotej.knez@frida:/shared/home/timotej.knez/llm-graph-construction
scp -r graph_building/llm timotej.knez@frida:/shared/home/timotej.knez/llm-graph-construction/graph_building
scp graph_building/graph_construction.py timotej.knez@frida:/shared/home/timotej.knez/llm-graph-construction/graph_building/graph_construction.py
scp graph_building/node_embeddings.py timotej.knez@frida:/shared/home/timotej.knez/llm-graph-construction/graph_building/node_embeddings.py
#scp -r dataset_loaders timotej.knez@frida:/shared/home/timotej.knez/llm-graph-construction
#scp -r dataLoaders timotej.knez@frida:/shared/home/timotej.knez/llm-graph-construction
scp -r custom_datasets timotej.knez@frida:/shared/home/timotej.knez/llm-graph-construction
scp -r models timotej.knez@frida:/shared/home/timotej.knez/llm-graph-construction
scp -r training timotej.knez@frida:/shared/home/timotej.knez/llm-graph-construction
scp -r train_graph.sbatch timotej.knez@frida:/shared/home/timotej.knez/llm-graph-construction
scp -r train_text.sbatch timotej.knez@frida:/shared/home/timotej.knez/llm-graph-construction
scp -r train_all.sbatch timotej.knez@frida:/shared/home/timotej.knez/llm-graph-construction
scp -r train_bimodal.sbatch timotej.knez@frida:/shared/home/timotej.knez/llm-graph-construction
scp -r precompute.sbatch timotej.knez@frida:/shared/home/timotej.knez/llm-graph-construction
scp -r precompute_graphs_for_analysis.sbatch timotej.knez@frida:/shared/home/timotej.knez/llm-graph-construction
rsync -av -r pipeline timotej.knez@frida:/shared/home/timotej.knez/llm-graph-construction
rsync -av -r eval.sbatch timotej.knez@frida:/shared/home/timotej.knez/llm-graph-construction

ssh frida "cd llm-graph-construction; sbatch --export=TP=wPrvGig2JoZB9uqQuFh4Kk3dD train_all.sbatch"