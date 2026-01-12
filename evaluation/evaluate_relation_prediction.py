import evaluate
import numpy as np
import torch
from torch_geometric.data import DataLoader
from transformers import TrainingArguments, Trainer

from custom_datasets.combining_data import window_row_entity_bert
from models.bimodal import MultiModalPrediction
from models.text_encoder import EntityBERTtextEncoder
from training.train_graph_encoder import prepare_dataset_combination_graph

def collate_function(examples):
    loader = DataLoader(examples, batch_size=len(examples))
    batch = next(iter(loader))
    return {"data": batch, "labels": batch.y}

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def window_text(graph):
    graph = window_row_entity_bert(graph, normalize_event_order=False)
    if graph is None:
        return None
    # classification_graph["text"], classification_graph["event1_start"], classification_graph["event1_end"], \
    # classification_graph["event2_start"], classification_graph["event2_end"] \
    #     = add_event_tokens(row["text"], row["event1_start"], row["event1_end"], row["event2_start"], row["event2_end"])
    return graph

def eval():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = torch.load("text-model.pt")
    model.to(device)
    dataset_train, dataset_val = prepare_dataset_combination_graph(balanced=False)
    # dataset_val = torch.load("pregenerated/dataset_small_for_experimenting.pt", map_location=device)
    # dataset_train.generated = list(filter(lambda x: x is not None, map(window_text, dataset_train.generated)))
    dataset_val.generated = list(filter(lambda x: x is not None, map(window_text, dataset_val.generated)))
    loader = DataLoader(dataset_val, batch_size=2)
    loader
    correct_sum = 0
    all_sum = 0
    for batch in loader:
        x = batch.to(device)
        y = batch.y.to(device)
        logits = model(x, y)["logits"]
        predictions = np.argmax(logits.detach().cpu(), axis=1)
        correct = sum(batch.y.cpu() == predictions)
        all = len(predictions)
        correct_sum += correct
        all_sum += all

    print("Accuracy:", correct_sum/all_sum)

if __name__ == '__main__':
    eval()