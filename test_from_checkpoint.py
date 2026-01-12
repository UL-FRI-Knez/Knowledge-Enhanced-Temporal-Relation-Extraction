import evaluate
import numpy as np
import torch
from torch_geometric.data import DataLoader
from transformers import trainer, Trainer, TrainingArguments

from custom_datasets.combining_data import read_i2b2
from custom_datasets.common import split_data
from custom_datasets.knowledge_graph_dataset import create_knowledge_graph_dataset, generate_relation_graph_llm
from models.bimodal import MultiModalPrediction
from models.knowledge_graph_encoder import GraphEncoder

def prepare_dataset_llm_only():
    df = torch.load("demo_dataset.pt")
    df_train, df_val, df_test = split_data(df, oversample=True, label_name='class', train_size=0.7, val_size=0.2, split_by_documents=False)
    dataset_train = create_knowledge_graph_dataset(df_train, generate_relation_graph_llm, cache_only=True)
    dataset_val = create_knowledge_graph_dataset(df_val, generate_relation_graph_llm, cache_only=True)
    dataset_train.pregenerate_and_filter()
    dataset_val.pregenerate_and_filter()
    return dataset_train, dataset_val

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def collate_function(examples):
    loader = DataLoader(examples, batch_size=len(examples))
    batch = next(iter(loader))
    return {"data": batch, "labels": batch.y}

dataset_train, dataset_val = prepare_dataset_llm_only()

# model = GraphEncoder.load_from_checkpoint("checkpoint-15000")
model = GraphEncoder(node_size=768, edge_size=768, number_of_relations=3, dropout=0.2)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-2,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    auto_find_batch_size=True,
    num_train_epochs=50,
    weight_decay=0.01,
    gradient_accumulation_steps=1,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    push_to_hub=False,
    resume_from_checkpoint="checkpoint-15000"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    data_collator=collate_function,
    compute_metrics=compute_metrics
)
print(trainer.predict(dataset_val))
