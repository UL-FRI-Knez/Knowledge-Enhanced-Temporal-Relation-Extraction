import gc
import os.path

from custom_datasets.negative_example_generation import add_null_relations

os.environ["WANDB_PROJECT"] = "relation-detection-extraction-i2b2"
from collections import Counter
from datetime import datetime
import numpy as np
import evaluate
import torch
from torch_geometric.data import DataLoader, Data
from transformers import Trainer, TrainingArguments

from custom_datasets.combining_data import window_row_entity_bert
from models.BaselineBERT import BaselineBERT
from models.GraphLanguageModel import GraphLanguageModel
from models.baselines.GPTmodel import GPTTemporalRelationExtraction
from models.bimodal import MultiModalPrediction
from models.knowledge_graph_encoder import GraphEncoder
from models.text_encoder import EntityBERTtextEncoder
from training.train_and_evaluate_relation_extraction import load_stored_dataset_combination_graph

results_file = "evaluation_results_no_relation/results-test-battery.txt"

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

def window_text(graph):
    graph = window_row_entity_bert(graph, normalize_event_order=False)
    if graph is None:
        return None
    return graph


device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"

def hyper_parameter_search(model_init, dataset_train, dataset_val, dataset_test, training_arguments, model_description):
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

    trainer = Trainer(
        model=None,
        args=training_arguments,
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
        n_trials=10,
        compute_objective=compute_objective,
    )
    return best_trial

def train_universal(model, dataset_steps, training_args_steps, model_description):
    model.to(device)
    datasets_created = False
    for i in range(len(dataset_steps)):
        if dataset_steps[i] is None:
            continue
        # free memory from previous step
        if datasets_created:
            del dataset_train
            del dataset_val
            del dataset_test
            gc.collect()
            datasets_created = False

        datasets_created = True
        if type(dataset_steps[i]) is tuple or type(dataset_steps[i]) is list:
            dataset_train, dataset_val, dataset_test = dataset_steps[i]
        else:
            print("running the dataset provider")
            dataset_train, dataset_val, dataset_test = dataset_steps[i]()
        dataset_train.generated = list(filter(lambda x: x is not None, map(window_text, dataset_train.generated)))
        dataset_val.generated = list(filter(lambda x: x is not None, map(window_text, dataset_val.generated)))
        dataset_test.generated = list(filter(lambda x: x is not None, map(window_text, dataset_test.generated)))

        training_args = training_args_steps[i]
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_train,
            eval_dataset=dataset_val,
            data_collator=collate_function,
            compute_metrics=compute_metrics
        )
        trainer.train()

        results = trainer.evaluate(eval_dataset=dataset_val)
        with open(results_file, "a") as myfile:
            myfile.write(model_description + " - midpoint results on eval:" + "\n")
            myfile.write(str(results) + "\n")
            myfile.flush()

    results = trainer.evaluate(eval_dataset=dataset_val)
    with open(results_file, "a") as myfile:
        myfile.write(model_description + " - End results - validation:" + "\n")
        myfile.write(str(results) + "\n")
        myfile.flush()

    results = trainer.evaluate(eval_dataset=dataset_test)
    with open(results_file, "a") as myfile:
        myfile.write(model_description + " - End results - test:" + "\n")
        myfile.write(str(results) + "\n")
        myfile.flush()

    del dataset_train
    del dataset_val
    del dataset_test
    gc.collect()
    return model

def train_graph(dataset_loader_balanced, dataset_loader_unbalanced, number_of_relations, test_name):
    model_bigger = GraphEncoder(node_size=768, edge_size=768 + 7, number_of_relations=number_of_relations, dropout=0.2)
    model = torch.load("best-models/graph_encoder-" + test_name + ".pt")
    model.post_mp = model_bigger.post_mp
    del model_bigger

    model.to(device)

    training_args = TrainingArguments(
        output_dir="./results-graph",
        learning_rate=2e-3,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        auto_find_batch_size=True,
        num_train_epochs=20,
        weight_decay=0.01,
        gradient_accumulation_steps=1,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        push_to_hub=False
    )

    model = train_universal(model,
                            [dataset_loader_balanced, dataset_loader_unbalanced],
                            [training_args, training_args], "Graph")

    torch.save(model, "evaluation_results_no_relation/graph_encoder-"+test_name+".pt")

    return model

def test_gpt_model(dataset_loader_balanced, dataset_loader_unbalanced,number_of_relations, test_name):
    dataset_train, dataset_val, _ = dataset_loader_balanced()
    model = GPTTemporalRelationExtraction()

    model.to(device)

    training_args = TrainingArguments(
        output_dir="./results-graph",
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
    results = trainer.evaluate(eval_dataset=dataset_val)
    with open(results_file, "a") as myfile:
        myfile.write("ChatGPT" + " - End results - val:" + "\n")
        myfile.write(str(results) + "\n")
        myfile.flush()

    del dataset_val
    gc.collect()

    _, _, dataset_test_ub = dataset_loader_unbalanced()
    results = trainer.evaluate(eval_dataset=dataset_test_ub)
    with open(results_file, "a") as myfile:
        myfile.write("ChatGPT" + " - End results - test:" + "\n")
        myfile.write(str(results) + "\n")
        myfile.flush()

    del dataset_test_ub
    gc.collect()
    return model


def test_baseline_model(dataset_loader_balanced, dataset_loader_unbalanced, test_name):
    _, dataset_val, _ = dataset_loader_balanced()
    dataset = dataset_val
    if dataset is not None:
        classes = [int(a.y) for a in dataset.generated]
        number_of_most_common_apperances = Counter(classes).most_common(1)[0][1]
        accuracy = number_of_most_common_apperances / len(dataset.generated)
        with open(results_file, "a") as myfile:
            myfile.write("Baseline" + " - balanced - val:" + "\n")
            myfile.write(str(accuracy) + "\n")
            myfile.flush()
    del dataset
    del dataset_val
    gc.collect()

    _, dataset_val_ub, dataset_test_ub = dataset_loader_unbalanced()
    dataset = dataset_val_ub
    if dataset is not None:
        classes = [int(a.y) for a in dataset.generated]
        number_of_most_common_apperances = Counter(classes).most_common(1)[0][1]
        accuracy = number_of_most_common_apperances / len(dataset.generated)
        with open(results_file, "a") as myfile:
            myfile.write("Baseline" + " - unbalanced - val:" + "\n")
            myfile.write(str(accuracy) + "\n")
            myfile.flush()

    del dataset
    del dataset_val_ub
    gc.collect()

    dataset = dataset_test_ub
    if dataset is not None:
        classes = [int(a.y) for a in dataset.generated]
        number_of_most_common_apperances = Counter(classes).most_common(1)[0][1]
        accuracy = number_of_most_common_apperances / len(dataset.generated)
        with open(results_file, "a") as myfile:
            myfile.write("Baseline" + " - unbalanced - test:" + "\n")
            myfile.write(str(accuracy) + "\n")
            myfile.flush()

    del dataset
    del dataset_test_ub
    gc.collect()

def train_glm(dataset_loader_balanced, dataset_loader_unbalanced, number_of_relations, test_name):
    model = GraphLanguageModel(number_of_relations=number_of_relations)

    training_args = TrainingArguments(
        output_dir="./results-glm",
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

    model = train_universal(model,
                            [dataset_loader_balanced, dataset_loader_unbalanced],
                            [training_args, training_args], "GLM model")

    torch.save(model, "evaluation_results_no_relation/gml-model-"+test_name+".pt")
    return model

def train_bimodal(dataset_loader_balanced, dataset_loader_unbalanced, graph_model, number_of_relations, test_name):
    model = MultiModalPrediction(number_of_relations=number_of_relations, combine_embeddings=True)
    model.graph_model = graph_model

    training_args = TrainingArguments(
        output_dir="./results-bimodal",
        # learning_rate=0.001,
        learning_rate=0.000955,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        auto_find_batch_size=True,
        num_train_epochs=50,
        # weight_decay=0.0001,
        weight_decay=0.000031,
        gradient_accumulation_steps=1,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        push_to_hub=False
    )

    model = train_universal(model,
                            [dataset_loader_balanced, dataset_loader_unbalanced],
                            [training_args, training_args], "Bimodal")

    torch.save(model, "evaluation_results_no_relation/bimodal-model-"+test_name+".pt")
    return model

def train_text(dataset_loader_balanced, dataset_loader_unbalanced, number_of_relations, test_name):
    model_bigger = EntityBERTtextEncoder(number_of_relations=number_of_relations, pooling_strategy='both_events')
    model = torch.load("best-models/graph_encoder-" + test_name + ".pt")
    model.post_layers = model_bigger.post_layers

    training_args = TrainingArguments(
        output_dir="./results-text",
        # learning_rate=0.0001,
        learning_rate=0.000955,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        auto_find_batch_size=True,
        num_train_epochs=50,
        # weight_decay=0.0001,
        weight_decay=0.000031,
        gradient_accumulation_steps=1,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        push_to_hub=False
    )
    model = train_universal(model,
                            [dataset_loader_balanced, dataset_loader_unbalanced],
                            [training_args, training_args], "Text")

    torch.save(model, "evaluation_results_no_relation/text-model-"+test_name+".pt")

    return model

def train_baseline_bert(dataset_loader_balanced, dataset_loader_unbalanced, number_of_relations, test_name):
    model = BaselineBERT(number_of_relations=number_of_relations, pooling_strategy='cls')

    training_args = TrainingArguments(
        output_dir="./results-baselineBERT",
        # learning_rate=0.0001,
        learning_rate=0.000955,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        auto_find_batch_size=True,
        num_train_epochs=50,
        # weight_decay=0.0001,
        weight_decay=0.000031,
        gradient_accumulation_steps=1,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        push_to_hub=False
    )

    # model = hyper_parameter_search(lambda _: BaselineBERT(number_of_relations=number_of_relations, pooling_strategy='cls'),
    #                                dataset_train, dataset_val, dataset_test_ub, training_args, "Baseline BERT (clinicalBERT)")
    model = train_universal(model,
                            [dataset_loader_balanced, dataset_loader_unbalanced],
                            [training_args, training_args], "Baseline BERT (clinicalBERT)")

    torch.save(model, "evaluation_results_no_relation/baseline-bert-model-hyper-"+test_name+".pt")

    return model

def full_testing_scenario(model, dataset_name, learning_rate, weight_decay):
    with open(results_file, "a") as myfile:
        myfile.write("\tmodel: " + str(model) + "\n")
        myfile.write("\tdataset: " + str(dataset_name) + "\n")
        myfile.write("\tlr: " + str(learning_rate) + "\n")
        myfile.write("\twd: " + str(weight_decay) + "\n")
        myfile.flush()


def predict():
    test_name = "i2b2"
    _, _, dataset_test = load_stored_dataset_combination_graph(balanced=False, dataset=test_name)
    model = torch.load("evaluation_results_no_relation/bimodal-model-" + test_name + ".pt")

    dataset_test.generated = list(filter(lambda x: x is not None, map(window_text, dataset_test.generated)))
    dataLoader = DataLoader(dataset_test, batch_size=1)
    for graph in dataLoader:
        prediction = model(graph, graph.y)

def train():
    test_name = "i2b2"
    print("Start training")
    with open(results_file, "a") as myfile:
        myfile.write("\nTest " + datetime.today().strftime('%Y-%m-%d %H:%M:%S') + "\n")
        myfile.write("\nDataset: " + test_name + "\n")
        myfile.flush()

    # dataset_loader_balanced = lambda: [add_null_relations(x) for x in load_stored_dataset_combination_graph(balanced=True, dataset=test_name)]
    dataset_loader_balanced = load_stored_dataset_combination_graph(balanced=True, dataset=test_name)
    dataset_loader_unbalanced = load_stored_dataset_combination_graph(balanced=False, dataset=test_name)
    # prune dataset to conserve memory

    for loader in [dataset_loader_balanced, dataset_loader_unbalanced]:
        for set in loader:
            new_generated = []
            for i in range(len(set.generated)):
                data = Data(
                    x=set.generated[i].x.detach().to(torch.float16),
                    y=set.generated[i].y,
                    edge_index=set.generated[i].edge_index,
                    edge_attr=set.generated[i].edge_attr.detach().to(torch.float16),
                    event1_index=set.generated[i].event1_index, event2_index=set.generated[i].event2_index,
                    text=set.generated[i].text,
                    event1_start=set.generated[i].event1_start,
                    event2_start=set.generated[i].event2_start,
                    event1_end=set.generated[i].event1_end,
                    event2_end=set.generated[i].event2_end)
                set.generated[i] = None
                new_generated.append(data)
            del set.generated
            set.generated = new_generated
            gc.collect()

        for set in loader:
            add_null_relations(set)

    if test_name == "thyme":
        number_of_relations = 11 # imamo relacije 0, 2, 3, 5, 6, 7, 8, 9, 10
    else:
        number_of_relations = 3

    # because we added the null relation
    number_of_relations += 1

    graph_model = train_graph(dataset_loader_balanced, dataset_loader_unbalanced, number_of_relations=number_of_relations, test_name=test_name)
    bimodal_model = train_bimodal(dataset_loader_balanced, dataset_loader_unbalanced, graph_model, number_of_relations=number_of_relations, test_name=test_name)
    text_model = train_text(dataset_loader_balanced, dataset_loader_unbalanced, number_of_relations=number_of_relations, test_name=test_name)
    # train_baseline_bert(dataset_loader_balanced, dataset_loader_unbalanced, number_of_relations=number_of_relations, test_name=test_name)
    # test_baseline_model(dataset_loader_balanced, dataset_loader_unbalanced, test_name=test_name)
    # test_gpt_model(dataset_loader_balanced, dataset_loader_unbalanced, number_of_relations=number_of_relations, test_name=test_name)
    # train_glm(dataset_loader_balanced, dataset_loader_unbalanced, number_of_relations=number_of_relations, test_name=test_name)

if __name__ == '__main__':
    train()