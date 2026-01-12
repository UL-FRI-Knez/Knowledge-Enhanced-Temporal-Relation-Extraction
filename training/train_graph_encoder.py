import os.path

import numpy as np
import evaluate
import torch
from torch_geometric.data import DataLoader
from transformers import Trainer, TrainingArguments

from custom_datasets.common import split_data, get_configuration_for_building_local_graph
from custom_datasets.combining_data import read_i2b2
from custom_datasets.dataframe_dataset import DFDataset
from custom_datasets.error_correction import fix_precomputed_dataset
from custom_datasets.knowledge_graph_dataset import create_knowledge_graph_dataset, \
    generate_llm_graph_for_event, generate_combination_graph, generate_relation_graph_llm
from graph_building.local_graph.build_local_patient_graph import construct_graph_from_text_only
from models.knowledge_graph_encoder import GraphEncoder

def prepare_dataset_llm_only():
    df = read_i2b2(full_text=True, use_test_files=False, include_rows_without_absolute=True)
    df_train, df_val, df_test = split_data(df, oversample=True, label_name='class', train_size=0.7, val_size=0.2, split_by_documents=True)
    dataset_train = create_knowledge_graph_dataset(df_train, generate_relation_graph_llm, cache_only=True)
    dataset_val = create_knowledge_graph_dataset(df_val, generate_relation_graph_llm, cache_only=True)
    dataset_train.pregenerate_and_filter()
    dataset_val.pregenerate_and_filter()
    return dataset_train, dataset_val

def prepare_dataset_combination_graph(balanced=True, return_val2=False):
    dataset_test = None
    if return_val2:
        if os.path.exists("pregenerated/dataset_test_fixed.pt"):
            dataset_test = torch.load("pregenerated/dataset_test_fixed.pt")
    if os.path.exists("pregenerated/dataset_train_fixed.pt") and os.path.exists("pregenerated/dataset_val_fixed.pt"):
        dataset_train = torch.load("pregenerated/dataset_train_fixed.pt")
        dataset_val = torch.load("pregenerated/dataset_val_fixed.pt")
        if not balanced:
            dataset_train.filter_out_repeated_entries()
            dataset_val.filter_out_repeated_entries()
        return dataset_train, dataset_val
    if os.path.exists("pregenerated/dataset_train.pt") and os.path.exists("pregenerated/dataset_val.pt"):
        dataset_train = DFDataset(save_path="pregenerated/dataset_train.pt")
        dataset_val = DFDataset(save_path="pregenerated/dataset_val.pt")
        print("Fixing precomputed train dataset")
        dataset_train = fix_precomputed_dataset(dataset_train)
        print("Fixing precomputed val dataset")
        dataset_val = fix_precomputed_dataset(dataset_val)
        torch.save(dataset_train, "pregenerated/dataset_train_fixed.pt")
        torch.save(dataset_val, "pregenerated/dataset_val_fixed.pt")
        if not balanced:
            dataset_train.filter_out_repeated_entries()
            dataset_val.filter_out_repeated_entries()
        return dataset_train, dataset_val
    df = read_i2b2(full_text=True, use_test_files=False, include_rows_without_absolute=True)
    df_train, df_val, df_test = split_data(df, oversample=True, label_name='class', train_size=0.7, val_size=0.2, split_by_documents=True)
    configuration = get_configuration_for_building_local_graph()

    if os.path.isfile("computed_kg.pt"):
        full_graph = torch.load("computed_kg.pt")
        train_document_ids = set(df_train["document_id"])
        val_document_ids = set(df_val["document_id"])
        patient_graphs_train = [x for x in full_graph if x[3] in train_document_ids]
        patient_graphs_val = [x for x in full_graph if x[3] in val_document_ids]
    else:
        patient_graphs_train = construct_graph_from_text_only(df_train, configuration, dataset_type="train")
        patient_graphs_val = construct_graph_from_text_only(df_val, configuration, dataset_type="val")

    dataset_train = create_knowledge_graph_dataset(df_train, generate_combination_graph, configuration=configuration, local_graph=patient_graphs_train, cache_only=True)
    dataset_val = create_knowledge_graph_dataset(df_val, generate_combination_graph, configuration=configuration, local_graph=patient_graphs_val, cache_only=True)
    dataset_train.pregenerate_and_filter()
    dataset_train.save("pregenerated/dataset_train.pt")
    dataset_val.pregenerate_and_filter()
    dataset_val.save("pregenerated/dataset_val.pt")

    if not balanced:
        dataset_train.filter_out_repeated_entries()
        dataset_val.filter_out_repeated_entries()

    if return_val2:
        return dataset_train, dataset_val, dataset_test
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
    return eval_pred["eval_accuracy"]

def train():

    # dataset_val = torch.load("pregenerated/dataset_small_for_experimenting.pt")
    # dataset_val.generated = [torch.load("pregenerated/primer_nepopravljenega_grafa.pt")]
    # fix_precomputed_dataset(dataset_val)

    dataset_train, dataset_val = prepare_dataset_combination_graph()
    # dataset_train, dataset_val = prepare_dataset_llm_only()

    model = GraphEncoder(node_size=768, edge_size=768 + 7, number_of_relations=3, dropout=0.2)
    # model = GraphEncoder(node_size=768, edge_size=768, number_of_relations=3, dropout=0.2)
    # model = MultiModalPrediction(number_of_relations=3, combine_embeddings=True)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-3,
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
    torch.save(model, "graph_encoder.pt")

def hyper_parameter_search():
    dataset_train, dataset_val = prepare_dataset_combination_graph()

    def model_init(trial):
        return GraphEncoder(node_size=768, edge_size=768 + 7, number_of_relations=3, dropout=0.2)

    def wandb_hp_space(trial):
        return {
            "name": "graphsweep-new",
            "method": "random",
            "metric": {"name": "validation_loss", "goal": "minimize"},
            "parameters": {
                "learning_rate": {"distribution": "uniform", "min": 1e-6, "max": 1e-1},
                "weight_decay": {"distribution": "uniform", "min": 1e-6, "max": 1e-1}
            },
        }

    training_args = TrainingArguments(
        output_dir="./results-graph",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        auto_find_batch_size=True,
        learning_rate=1e-2,
        weight_decay=1e-4,
        num_train_epochs=50,
        gradient_accumulation_steps=1,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_steps=5000,
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
        n_trials=100,
        # compute_objective=compute_objective,
    )
    return best_trial

def test_model():
    pass

if __name__ == '__main__':
    train()