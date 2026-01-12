# Knowledge Graph-Enhanced Temporal Relation Extraction

This repository contains the code for research on using external knowledge to support temporal relation extraction in medical documents. The project implements both a basic model and an end-to-end pipeline for extracting temporal relations from clinical text.

## Overview

The system combines information from text with information from knowledge graphs to produce accurate temporal relation predictions. The approach is particularly suited for medical documents where temporal ordering of events (e.g., symptoms, treatments, diagnoses) is crucial for understanding patient narratives.

### Key Features

- **Bimodal Learning**: Combines textual information with knowledge graph embeddings
- **End-to-End Pipeline**: Performs event extraction, event pair generation, and temporal relation classification
- **Knowledge Graph Integration**: Leverages external medical knowledge (PrimeKG) and LLM-generated knowledge
- **Memory-Augmented Predictions**: Stores extracted temporal relations to improve future predictions
- **Multiple Datasets**: Supports i2b2 and THYME temporal relation datasets

## Architecture

### Basic Model Components

1. **Text Encoder** (`models/text_encoder.py`): Encodes event mentions and their context using transformer-based models
2. **Knowledge Graph Encoder** (`models/knowledge_graph_encoder.py`): Encodes relevant subgraphs from external knowledge sources
3. **Bimodal Fusion** (`models/bimodal.py`): Combines text and graph representations for relation prediction

### Pipeline Components

The end-to-end pipeline (`pipeline/`) performs:
1. **Event Extraction**: Identifies medical events in text using trained sequence tagging models
2. **Event Pair Generation**: Creates candidate event pairs for temporal relation classification
3. **Relation Extraction**: Uses the pretrained bimodal model to classify temporal relations
4. **Memory Storage**: Stores extracted relations to support future predictions

## Project Structure

```
.
├── training/                          # Training scripts for all models
│   ├── train_text_encoder.py         # Train text encoder
│   ├── train_graph_encoder.py        # Train graph encoder
│   ├── train_combined_relation_encoder.py  # Train combined model
│   ├── train_event_extraction.py     # Train event extraction model
│   └── train_and_evaluate_relation_extraction.py  # Full training pipeline
├── pipeline/                          # End-to-end pipeline implementation
│   ├── pipeline.py                    # Main pipeline code
│   └── pipeline_evaluation.py        # Pipeline evaluation utilities
├── models/                            # Model architectures
│   ├── text_encoder.py               # Text encoding models
│   ├── knowledge_graph_encoder.py    # Graph encoding models
│   ├── bimodal.py                    # Multimodal fusion model
│   └── baselines/                    # Baseline implementations
├── custom_datasets/                   # Data loading and preprocessing
│   ├── i2b2dataLoader.py             # i2b2 dataset loader
│   ├── thyme_loader.py               # THYME dataset loader
│   ├── knowledge_graph_dataset.py    # KG dataset construction
│   └── event_extraction_dataset.py   # Event extraction data
├── graph_building/                    # Knowledge graph construction
├── evaluation/                        # Evaluation scripts
├── notebooks/                         # Analysis and experimentation notebooks
│   ├── Event extraction evaluation.ipynb
│   ├── Error analysis.ipynb
│   ├── End-to-end ablation study.ipynb
│   └── Manual end-to-end evaluation.ipynb
├── batch_scripts/                     # SLURM batch scripts for HPC
├── data/                             # Dataset storage (not included)
│   ├── i2b2/                         # i2b2 temporal relation corpus
│   ├── thyme/                        # THYME corpus
│   └── primekg.tab                   # PrimeKG knowledge graph
└── best-models/                      # Saved model checkpoints
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- Transformers (HuggingFace)
- Flair NLP
- Additional dependencies in `wandb/*/files/requirements.txt`

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd KG-temporal-relation-extraction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (adjust based on your requirements.txt)
pip install torch torchvision torchaudio
pip install torch-geometric
pip install transformers flair
pip install wandb pandas numpy scikit-learn
```

## Usage

### Training Models

#### 1. Train Text Encoder
```bash
python -m training.train_text_encoder
```

#### 2. Train Graph Encoder
```bash
python -m training.train_graph_encoder
```

#### 3. Train Combined Relation Extraction Model
```bash
python -m training.train_combined_relation_encoder
```

#### 4. Train Event Extraction Model
```bash
python -m training.train_event_extraction
```

### Running the End-to-End Pipeline

```python
from pipeline import pipeline

# Load your clinical documents
# Run the pipeline for event extraction and relation classification
results = pipeline.run()
```

### Evaluation

Evaluation scripts are available in the `evaluation/` folder:

```bash
python evaluation/evaluate_relation_prediction.py
```

### Using Notebooks

The `notebooks/` folder contains Jupyter notebooks for various analyses:

- **Event extraction evaluation.ipynb**: Evaluate event extraction performance
- **Error analysis.ipynb**: Analyze model errors and failure cases
- **End-to-end ablation study.ipynb**: Ablation studies on pipeline components
- **Manual end-to-end evaluation.ipynb**: Manual evaluation of pipeline outputs

## Datasets

This project supports two main temporal relation datasets:

1. **i2b2 2012 Temporal Relations Challenge**: Clinical notes with temporal relation annotations
2. **THYME Corpus**: Cancer pathology reports with rich temporal annotations

The knowledge graph component uses:
- **PrimeKG**: A precision medicine knowledge graph
- **LLM-generated knowledge**: On-demand knowledge extraction using language models

### Data Preparation

Place your datasets in the `data/` directory:
```
data/
├── i2b2/           # i2b2 XML files
├── i2b2-test/      # i2b2 test set
└── primekg.tab     # PrimeKG file
```

## Configuration

Key configuration parameters can be set in `custom_datasets/common.py` through the `Configuration` class:

- `add_inverse_relations_to_graph`: Include inverse relations in KG
- `use_realistic_graph`: Use realistic graph construction (limited to available knowledge)
- `remove_target_relation`: Remove gold relations from graph during training

## Acknowledgments

This research uses:
- The i2b2 2012 Temporal Relations corpus
- The THYME corpus
- PrimeKG knowledge graph
- HuggingFace Transformers
- PyTorch Geometric
- Flair NLP

