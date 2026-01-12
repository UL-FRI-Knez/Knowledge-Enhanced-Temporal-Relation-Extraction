import evaluate
import numpy as np
import torch

from torch_geometric.data import DataLoader, Data
from transformers import Trainer, TrainingArguments

from custom_datasets import combining_data
from custom_datasets.combining_data import read_i2b2, normalize_event_order
from custom_datasets.common import split_data
from custom_datasets.knowledge_graph_dataset import create_knowledge_graph_dataset, generate_llm_graph_for_event, \
    generate_relation_graph_llm, relation_types
from models.knowledge_graph_encoder import GraphEncoder
from models.text_encoder import EntityBERTtextEncoder
from training.train_graph_encoder import prepare_dataset_combination_graph


def prepare_dataset_llm_only():
    df = read_i2b2(full_text=True, use_test_files=False, include_rows_without_absolute=True)
    df_train, df_val, df_test = split_data(df, oversample=True, label_name='class', train_size=0.7, val_size=0.2, split_by_documents=True)
    dataset_train = create_knowledge_graph_dataset(df_train, generate_relation_graph_llm, cache_only=True)
    dataset_val = create_knowledge_graph_dataset(df_val, generate_relation_graph_llm, cache_only=True)
    dataset_train.pregenerate_and_filter()
    dataset_val.pregenerate_and_filter()
    return dataset_train, dataset_val

def prepare_dataset_no_graph(oversample=False):
    def get_empty_graph(row, **kwargs):
        target = row["class"]
        return Data(x=torch.empty((0,0)), y=torch.tensor([relation_types.index(target)]), edge_index=torch.empty((0,0)), edge_attr=torch.empty((0,0)),
                    event1_index=0, event2_index=0)
    df = read_i2b2(full_text=True, use_test_files=False, include_rows_without_absolute=True)
    df = normalize_event_order(df)
    df_train, df_val, df_test = split_data(df, oversample=oversample, label_name='class', train_size=0.7, val_size=0.2, split_by_documents=True)
    dataset_train = create_knowledge_graph_dataset(df_train, get_empty_graph, cache_only=True)
    dataset_val = create_knowledge_graph_dataset(df_val, get_empty_graph, cache_only=True)
    dataset_train.pregenerate_and_filter()
    dataset_val.pregenerate_and_filter()
    return dataset_train, dataset_val

def collate_function(examples):
    loader = DataLoader(examples, batch_size=len(examples))
    batch = next(iter(loader))
    return {"data": batch, "labels": batch.y}

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def compute_objective(eval_pred):
    return eval_pred["eval_loss"]

def train():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)

    model = EntityBERTtextEncoder(number_of_relations=3)

    model2 = torch.load("pretrained models/EntityBert_relation_extraction.pt")
    model.EntityBert = model2.EntityBert

    model.to(device)

    dataset_train, dataset_val = prepare_dataset_no_graph(oversample=True)
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=0.1,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        auto_find_batch_size=True,
        num_train_epochs=50,
        weight_decay=0.01,
        gradient_accumulation_steps=1,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        push_to_hub=False
    )

    # "adamw_hf", "adamw_torch", "adamw_torch_fused", "adamw_apex_fused", "adamw_anyprecision" or "adafactor"
    # training_args.set_optimizer(name="adafactor")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        data_collator=collate_function,
        compute_metrics=compute_metrics
    )
    trainer.train()


    dataset_train, dataset_val = prepare_dataset_no_graph(oversample=False)
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=0.1,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        auto_find_batch_size=True,
        num_train_epochs=50,
        weight_decay=0.01,
        gradient_accumulation_steps=1,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        push_to_hub=False
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        data_collator=collate_function,
        compute_metrics=compute_metrics
    )
    trainer.train()
    torch.save(model, "text-model.pt")


def hyper_parameter_search():
    dataset_train, dataset_val = prepare_dataset_no_graph(oversample=True)

    def model_init(trial):
        return EntityBERTtextEncoder(number_of_relations=3)

    def wandb_hp_space(trial):
        return {
            "name": "textsweep",
            "method": "random",
            "metric": {"name": "validation_loss", "goal": "minimize"},
            "parameters": {
                "learning_rate": {"distribution": "uniform", "min": 1e-5, "max": 1e-2},
                "weight_decay": {"distribution": "uniform", "min": 1e-5, "max": 1e-1}
            },
        }

    training_args = TrainingArguments(
        output_dir="./results_text",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        auto_find_batch_size=True,
        learning_rate=2e-4,
        weight_decay=0.0000001,
        num_train_epochs=50,
        gradient_accumulation_steps=1,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_steps=1000,
        push_to_hub=False
    )

    trainer = Trainer(
        model=None,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        compute_metrics=compute_metrics,
        model_init=model_init,
        data_collator=collate_function,
    )

    best_trial = trainer.hyperparameter_search(
        direction="minimize",
        backend="wandb",
        hp_space=wandb_hp_space,
        n_trials=40,
        compute_objective=compute_objective,
    )
    torch.save(best_trial, "text-best-trail.pt")
    return best_trial

if __name__ == '__main__':
    hyper_parameter_search()