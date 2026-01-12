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

def train():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    # model = MultiModalPrediction()
    # # text_model = torch.load("text-model.pt")
    # graph_model = torch.load("graph_encoder.pt", map_location=device)
    # # model.text_model = text_model
    # model.graph_model = graph_model
    # model = torch.load("multimodal-model-balanced.pt", map_location=device)

    model = torch.load("text-model-balanced.pt")
    # model = EntityBERTtextEncoder()

    model.to(device)
    # model = MultiModalPrediction(number_of_relations=3, combine_embeddings=True)

    dataset_train, dataset_val = prepare_dataset_combination_graph(balanced=True)
    # dataset_train = torch.load("pregenerated/dataset_small_for_experimenting.pt", map_location=device)
    # dataset_val = torch.load("pregenerated/dataset_small_for_experimenting.pt", map_location=device)
    # dataset_train = torch.load("demo_dataset.pt")
    dataset_train.generated = list(filter(lambda x: x is not None, map(window_text, dataset_train.generated)))
    dataset_val.generated = list(filter(lambda x: x is not None, map(window_text, dataset_val.generated)))

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=0.001,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        auto_find_batch_size=True,
        num_train_epochs=50,
        weight_decay=0.0001,
        gradient_accumulation_steps=1,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        push_to_hub=False
    )
    # "adamw_hf", "adamw_torch", "adamw_torch_fused", "adamw_apex_fused", "adamw_anyprecision" or "adafactor"
    training_args.set_optimizer(name="adafactor")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        data_collator=collate_function,
        compute_metrics=compute_metrics
    )
    trainer.train()

    # torch.save(model, "text-model-balanced.pt")

    torch.save(model, "text-model-2.pt")

if __name__ == '__main__':
    train()