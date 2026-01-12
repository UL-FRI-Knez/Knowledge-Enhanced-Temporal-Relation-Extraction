import argparse
import concurrent.futures

import torch
import sys

import evaluation.evaluate_relation_prediction
from custom_datasets.combining_data import read_i2b2
from custom_datasets.common import Configuration, lock
from custom_datasets.knowledge_graph_dataset import get_llm_responses_only, generate_relation_graph_llm, \
    generate_relation_graph_primekg
from graph_building.local_graph.build_local_patient_graph import construct_graph_from_text_only
from pipeline import pipeline
from training import train_text_encoder, train_graph_encoder, train_combined_relation_encoder, \
    train_and_evaluate_relation_extraction, train_and_evaluate_relation_detection_extraction
from training.train_graph_encoder import hyper_parameter_search, train

def prepare_llm_responses():
    df = read_i2b2(full_text=True, use_test_files=False, include_rows_without_absolute=True)
    get_llm_responses_only(df)

def run_graph_encoder_optimization():
    best_trial = hyper_parameter_search()
    torch.save(best_trial, "best_trial.pt")
    pass

def precompute_local_graphs():
    configuration = Configuration()
    configuration.add_inverse_relations_to_graph = True
    configuration.remove_target_relation = False
    configuration.use_realistic_graph = True
    df = read_i2b2(full_text=True, use_test_files=False, include_rows_without_absolute=True)
    in_memory_kg = construct_graph_from_text_only(df, configuration, dataset_type="train")
    torch.save(in_memory_kg, "computed_kg.pt")

def precompute_graphs_for_analysis():
    df = read_i2b2(full_text=True, use_test_files=False, include_rows_without_absolute=True)
    generated = [None for _ in df.iloc]
    def row_converter(row, args):
        llm_kg = generate_relation_graph_llm(row, cache_only=True)
        primekg_kg = generate_relation_graph_primekg(row)
        return(llm_kg, primekg_kg)

    environment = {"generated": generated, "df": df.iloc, "row_converter": row_converter, "args": {}, "finished": 0}
    def pretvori(ind, environment):
        environment["generated"][ind] = environment["row_converter"](environment["df"][ind], environment["args"])
        with lock:
            environment["finished"] += 1
            if environment["finished"] % 100 == 0:
                torch.save(environment["generated"], "graphs_for_analysis.pt")
        print("Graph", ind, "generated")


    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        print(f"\n ... executing workers ...\n")
        for i in range(len(df)):
            executor.submit(pretvori, i, environment)


if __name__ == '__main__':
    import wandb
    wandb.login(key="4627d8b8a181a377246f040719f6769bc68465fa")
    import nltk
    nltk.download('punkt_tab')

    parser = argparse.ArgumentParser(description="sample argument parser")
    parser.add_argument("--method", default="prepare_llm_responses")
    parser.add_argument("--event_detector", default="bert")
    parser.add_argument("--event_pairs", default="all")
    args = parser.parse_args()
    if args.method == "prepare_llm_responses":
        prepare_llm_responses()
    elif args.method == "train_graph":
        train_graph_encoder.train()
    elif args.method == "train_text":
        train_text_encoder.hyper_parameter_search()
    elif args.method == "train_all":
        train_and_evaluate_relation_extraction.train()
    elif args.method == "precompute_local_graphs":
        precompute_local_graphs()
    elif args.method == "precompute_graphs_for_analysis":
        precompute_graphs_for_analysis()
    elif args.method == "train_bimodal":
        train_combined_relation_encoder.train()
    elif args.method == "eval":
        evaluation.evaluate_relation_prediction.eval()
    elif args.method == "eval_pipeline":
        pipeline.run_pipeline(args)
    elif args.method == "train_relation_detection_and_prediction":
        train_and_evaluate_relation_detection_extraction.train()
