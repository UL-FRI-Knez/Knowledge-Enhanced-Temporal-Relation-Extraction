import json
import os
from typing import Literal

import spacy
from collections import Counter

import nltk as nltk
import pandas as pd
import torch
from torch_geometric.data import DataLoader, Data
from transformers import AutoTokenizer
import numpy as np
import nltk.data

from custom_datasets.combining_data import read_i2b2
from custom_datasets.common import Configuration, get_configuration_for_building_local_graph
from custom_datasets.error_correction import generate_edge_embedding
from custom_datasets.knowledge_graph_dataset import generate_relation_graph_llm, generate_relation_graph_primekg, \
    generate_local_graph_for_event, create_knowledge_graph_dataset, generate_combination_graph, \
    generate_fast_combination_graph
from custom_datasets.dataframe_dataset import DFDataset
from graph_building import node_embeddings
from graph_building.local_graph.build_local_patient_graph import construct_graph_from_text_only
from flair.models import SequenceTagger
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Sentence

test_name = "i2b2"


def get_flert_predictions(tagged_sentence, original_text):
    predicted_events = []
    curent_event = [0, 0]
    in_event = False
    offset = 0
    for word in tagged_sentence:
        offset = original_text.find(word.text, offset)
        if word.tag == "B":
            if in_event:
                # complete curent event
                predicted_events.append(
                    (curent_event[0], curent_event[1], original_text[curent_event[0]: curent_event[1]]))
                current_event = [offset, offset]
                in_event = False
            in_event = True
            curent_event[0] = offset
            curent_event[1] = offset + len(word.text)
        elif word.tag == "I" and in_event == True:
            curent_event[1] = offset + len(word.text)
        elif word.tag == "I" and in_event == False:
            # ignore events without a beginning
            # in_event = True
            # curent_event[0] = offset
            # curent_event[1] = offset + len(word.text)
            pass
        elif word.tag == "X" and in_event == True:
            # complete curent event
            predicted_events.append(
                (curent_event[0], curent_event[1], original_text[curent_event[0]: curent_event[1]]))
            curent_event = [offset, offset]
            in_event = False

        offset += len(word.text)
    return predicted_events


if test_name == "thyme":
    flert_model = SequenceTagger.load('resources/taggers/sota-ner-flert-' + test_name + '/final-model.pt')
else:
    flert_model = SequenceTagger.load('resources/taggers/sota-ner-flert-' + test_name + '2/final-model.pt')

def extract_events_flert(text):
    sentence = Sentence(text)
    # predict tags and print
    flert_model.predict(sentence)
    predictions = get_flert_predictions(sentence, text)
    return predictions

def extract_events(text):
    event_extraction_model = torch.load("event-model.pt", map_location=torch.device('cpu'))
    sentence_idxs = split_text_into_sentenes(text)
    events = []
    for sentence_idx in sentence_idxs:
        tokenized = event_extraction_model.tokenizer(text[sentence_idx[0]:sentence_idx[1]], return_tensors="pt")
        classification = event_extraction_model(tokenized)
        predictions = torch.argmax(classification["results"], axis=1)
        start = 0
        end = 0
        inside_event = False
        for i, prediction in enumerate(predictions):
            if prediction == 1:
                if tokenized.token_to_chars(i) is None:
                    continue
                if not inside_event:
                    start = tokenized.token_to_chars(i).start
                end = tokenized.token_to_chars(i).end
                inside_event = True
            else:
                if inside_event:
                    # end this event
                    events.append((sentence_idx[0] + start, sentence_idx[0] + end, text[sentence_idx[0] + start:sentence_idx[0] + end]))
                inside_event = False
        if inside_event:
            # end this event
            events.append((sentence_idx[0] + start, sentence_idx[0] + end, text[sentence_idx[0] + start:sentence_idx[0] + end]))
    return events

def split_text_into_sentenes(text):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = list(tokenizer.span_tokenize(text))
    return sentences

def generate_all_event_pairs(events):
    pairs = []
    for i in range(len(events) - 1):
        for j in range(i+1, len(events)):
            pairs.append((events[i], events[j]))
    return pairs

nlp = spacy.load("en_core_web_sm")
def generate_event_pairs(text, events):
    pairs = []
    # get event pars that appear in the same sentence
    sentences = split_text_into_sentenes(text)
    event_sentence_indexes = []
    for event_idx, event in enumerate(events):
        start, end, event_text = event
        sentence_index = get_sentence_index(sentences, start, end)
        event_sentence_indexes.append((sentence_index, event_idx))

    for sentence_idx in range(len(sentences)):
        events_from_this_sentence = list(map(lambda event_sent: events[event_sent[1]], filter(lambda event_sent: event_sent[0] == sentence_idx, event_sentence_indexes)))
        pairs += generate_all_event_pairs(events_from_this_sentence)

    # between sentence pairs
    # split events into sentences
    events_in_sentences = [[] for _ in sentences]
    for event_idx, event in enumerate(events):
        start, end, event_text = event
        sentence_index = get_sentence_index(sentences, start, end)
        events_in_sentences[sentence_index].append((event, event_idx))
    # sort event in sentences
    for s in events_in_sentences:
        s.sort(key=lambda event: event[0][0])
    # add relations between main events in subsequent sentences
    for i in range(len(events_in_sentences)-1):
        if len(events_in_sentences[i]) == 0 or len(events_in_sentences[i+1]) == 0:
            continue
        pairs.append((events_in_sentences[i][0][0], events_in_sentences[i + 1][0][0]))
        if len(events_in_sentences[i]) > 1:
            pairs.append((events_in_sentences[i][-1][0], events_in_sentences[i+1][0][0]))
        if len(events_in_sentences[i+1]) > 1:
            pairs.append((events_in_sentences[i][0][0], events_in_sentences[i + 1][-1][0]))
        if len(events_in_sentences[i]) > 1 and len(events_in_sentences[i+1]) > 1:
            pairs.append((events_in_sentences[i][-1][0], events_in_sentences[i + 1][-1][0]))

    # Add relations between events with correference
    events_by_main_words = {}
    for event in events:
        head_word = ""
        for token in nlp(event[2]):
            if token.dep_ == "ROOT":
                head_word = token.text
        if head_word not in events_by_main_words:
            events_by_main_words[head_word] = []
        events_by_main_words[head_word].append(event)
    for hw in events_by_main_words:
        evs = events_by_main_words[hw]
        for i in range(len(evs)-1):
            for j in range(i+1,len(evs)):
                pairs.append((evs[i], evs[j]))
    return pairs

def get_sentence_index(sentences, start, end):
    sentence_index = 0
    for sentence_idx, (sent_start, sent_end) in enumerate(sentences):
        if sent_start <= start and sent_end >= end:
            sentence_index = sentence_idx
    return sentence_index


def construct_basic_dataframe(text, pairs, document_id):
    rows = []
    for pair in pairs:
        row = [
            text,
            "BEFORE",
            pair[0][0],
            pair[0][1],
            None,
            pair[1][0],
            pair[1][1],
            None,
            pair[0][2],
            pair[1][2],
            document_id,
            "eval",
            None
        ]
        rows.append(row)
    return pd.DataFrame(rows, columns=['text', 'class', 'event1_start', 'event1_end', 'event1_type', 'event2_start', 'event2_end', 'event2_type', 'event1_text', 'event2_text', 'document_id', 'source', 'additional_document_info'])
    pass


def construct_dataset_with_graphs(text, dataframe, patient_id):
    configuration = get_configuration_for_building_local_graph()
    local_kg = construct_graph_from_text_only(dataframe, configuration, dataset_type="train")
    dataset = create_knowledge_graph_dataset(dataframe, generate_fast_combination_graph, configuration=configuration,
                                    local_graph=local_kg, cache_only=False, insert_time_nodes=True,
                                    graph_post_processing=add_stored_data_to_kg, patient_id=patient_id, relation_types=relation_types)
    dataset.pregenerate_and_filter()
    print(dataset)
    return dataset

def construct_dataset_no_graph(text, dataframe, patient_id):
    def get_empty_graph(row, **kwargs):
        target = row["class"]
        return Data(x=torch.empty((0,0)), y=torch.tensor([relation_types.index(target)]), edge_index=torch.empty((0,0)), edge_attr=torch.empty((0,0)),
                    event1_index=0, event2_index=0)
    dataset = create_knowledge_graph_dataset(dataframe, get_empty_graph, cache_only=True)
    return dataset

relation_types = ["BEFORE", "AFTER", "OVERLAP"]
def predict_temporal_relations(dataset, relation_detection_method:Literal["none", "shared", "separate"]="none", args=None):
    human_readable_predictions = []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if relation_detection_method == "separate":
        relation_detection_model = torch.load("best-models/relation-detection-model.pt", map_location=device)
    if relation_detection_method == "shared":
        model = torch.load("best-models/bimodal-model-i2b2-norelation.pt", map_location=device)
    else:
        if args.relation_model == "bimodal":
            model = torch.load("best-models/bimodal-model-i2b2.pt", map_location=device)
        elif args.relation_model == "bimodal_transitive":
            model = torch.load("old-models/Transitive_relation_extraction.pt", map_location=device)

    loader = DataLoader(dataset, batch_size=16)
    for batch in loader:
        batch.to(device)
        labels = batch.y
        if relation_detection_method == "separate":
            relation_detection = relation_detection_model(data=batch, labels=labels)
            predictions_relation_detection = np.argmax(relation_detection["logits"].detach().cpu(), axis=1)
            pass
        if args.relation_model == "bimodal_transitive":
            result = model(data=batch, return_embedding=False)
            predictions = np.argmax(result.cpu().detach().numpy(), axis=1)
            confidence = [[float(x) for x in p] for p in result.cpu().detach().numpy()]
        else:
            result = model(data=batch, labels=labels)
            predictions = np.argmax(result["predictions"].cpu().detach().numpy(), axis=1)
            confidence = [[float(x) for x in p] for p in result["predictions"].cpu().detach().numpy()]
        for i in range(len(batch["text"])):
            event1 = batch["event1_oroginal_position"][i]
            event2 = batch["event2_oroginal_position"][i]
            if predictions[i] < len(relation_types):
                if relation_detection_method == "separate":
                    human_readable_predictions.append((event1, relation_types[predictions[i]], event2, predictions_relation_detection[i], confidence[i]))
                else:
                    human_readable_predictions.append((event1, relation_types[predictions[i]], event2, confidence[i]))
    return human_readable_predictions

patient_graphs = {}
global_graph = {}
def compute_event_pair_key(event1, event2):
    return (event1, event2)
def add_stored_data_to_kg(graph, patient_id, **kwargs):
    global patient_graphs, global_graph
    event1 = graph.text[graph.event1_start: graph.event1_end]
    event2 = graph.text[graph.event2_start: graph.event2_end]
    event_pair_key = compute_event_pair_key(event1, event2)
    edge_index_list = graph.edge_index.tolist()
    edge_attr_list = graph.edge_attr.tolist()
    if isinstance(graph.edge_type, list):
        edge_type_list = graph.edge_type
    else:
        edge_type_list = graph.edge_type.tolist()
    if patient_id in patient_graphs:
        patient_graph = patient_graphs[patient_id]
    else:
        patient_graph = {}
    if event_pair_key in global_graph:
        link_probabilities = [global_graph[event_pair_key]["BEFORE"],
                              global_graph[event_pair_key]["AFTER"],
                              global_graph[event_pair_key]["OVERLAP"]]
        link_probabilities = torch.tensor(link_probabilities)
        link_probabilities = link_probabilities / sum(link_probabilities)
        link_embedding = generate_edge_embedding("temporal_relation", 1, link_probabilities)
        edge_index_list[0].append(graph.event1_index)
        edge_index_list[1].append(graph.event2_index)
        edge_attr_list.append(link_embedding)
        edge_type_list.append(1)

    if event_pair_key in patient_graph:
        link_probabilities = [patient_graph[event_pair_key]["BEFORE"],
                              patient_graph[event_pair_key]["AFTER"],
                              patient_graph[event_pair_key]["OVERLAP"]]
        link_probabilities = torch.tensor(link_probabilities)
        link_probabilities /= sum(link_probabilities)
        link_embedding = generate_edge_embedding("temporal_relation", 1, link_probabilities)
        edge_index_list[0].append(graph.event1_index)
        edge_index_list[1].append(graph.event2_index)
        edge_attr_list.append(link_embedding)
        edge_type_list.append(1)
    graph.edge_index = torch.tensor(edge_index_list)
    graph.edge_attr = torch.tensor(edge_attr_list)
    graph.edge_type = torch.tensor(edge_type_list)
    return graph

def save_to_common_graph(predicted_relations, patient_id):
    global patient_graphs, global_graph
    if patient_id in patient_graphs:
        patient_graph = patient_graphs[patient_id]
    else:
        patient_graph = {}
    for relation in predicted_relations:
        events_key = compute_event_pair_key(relation[0], relation[2])
        if events_key not in patient_graph:
            patient_graph[events_key] = {"counter": Counter(), "event_mentions": (relation[0], relation[2])}
        patient_graph[events_key]["counter"].update([relation[1]])

        if events_key not in global_graph:
            global_graph[events_key] = {"counter": Counter(), "event_mentions": (relation[0], relation[2])}
        global_graph[events_key]["counter"].update([relation[1]])
    patient_graphs[patient_id] = patient_graph

def get_sentences_from_test_set_i2b2():
    all_document_ids = []
    all_documents = []
    all_sentences = []
    all_relations = []
    i2b2df = read_i2b2(full_text=True, use_test_files=True, include_rows_without_absolute=True)
    documents = set(i2b2df["document_id"])
    for document in documents:
        all_document_ids.append(document)
        relations = i2b2df[i2b2df["document_id"] == document].reset_index(drop=True)
        text = relations["text"][0]
        all_documents.append(text)
        sentences = split_text_into_sentenes(text)
        sentences_text = [text[s[0]: s[1]] for s in sentences]
        all_sentences.append(sentences_text)
        relations_in_document = []
        for i, relation in relations.iterrows():
            event1 = (relation["event1_start"], relation["event1_end"], relation["event1_text"])
            event2 = (relation["event2_start"], relation["event2_end"], relation["event2_text"])
            relations_in_document.append((event1, relation["class"], event2))
        all_relations.append(relations_in_document)
    return all_documents, all_sentences, all_relations, all_document_ids

def overlap(event1, event2):
    start1, end1, text1 = event1
    start2, end2, text2 = event2
    if start1 >= start2 and start1 <= end2:
        # print(event1, event2)
        return True
    if end1 >= start2 and end1 <= end2:
        # print(event1, event2)
        return True
    if start2 >= start1 and start2 <= end1:
        # print(event1, event2)
        return True
    return False

def most_similar_event_expanded(event_list :list[tuple[int, int, str]], event_target):
    # min_difference = -1
    # best_match = None
    # for e in event_list:
    #     dif = -1
    #     event_text = e[2]
    #     if event_text in event_target[2]:
    #         dif = len(event_target[2]) - len(event_text)
    #     if event_target[2] in event_text:
    #         dif = len(event_text) - len(event_target[2])
    #     if dif >= 0 and (min_difference < 0 or min_difference > dif):
    #         min_difference = dif
    #         best_match = e
    # return best_match
    for e in event_list:
        if overlap(e, event_target):
            return e
    return None

def analyze_document(text, patient_id, args, event_pairs_of_interest=None, relation_detection_method="separate", document_id=""):
    if len(document_id) == 0:
        document_id = str(hash(text))
    if os.path.isfile("pipeline_tmp_"+args.event_detector + "_" + args.event_pairs + "/" + document_id + ".pt"):
        dataset = DFDataset()
        dataset.load("pipeline_tmp_"+args.event_detector + "_" + args.event_pairs + "/" + document_id + ".pt")
    else:
        if args.event_detector == "bert":
            events = extract_events(text)
        elif args.event_detector == "flert":
            events = extract_events_flert(text)
        print("Events:")
        print(events)
        if event_pairs_of_interest is None:
            event_pairs = generate_event_pairs(text, events)
        else:
            gold_event_pairs = []
            for e1, e2 in event_pairs_of_interest:
                e1 = most_similar_event_expanded(events, e1)
                e2 = most_similar_event_expanded(events, e2)
                if e1 is not None and e2 is not None:
                    gold_event_pairs.append((e1, e2))
            if args.event_pairs == "gold_only":
                event_pairs = []
            else:
                event_pairs = generate_event_pairs(text, events)
            
        dataframe = construct_basic_dataframe(text, gold_event_pairs + event_pairs, 0)
        dataset = construct_dataset_with_graphs(text, dataframe, patient_id)
        for i in range(len(dataset.generated)):
            dataset.generated[i].gold_relations = (i < len(gold_event_pairs))
        if not os.path.isdir("pipeline_tmp_"+args.event_detector + "_" + args.event_pairs):
            os.mkdir("pipeline_tmp_"+args.event_detector + "_" + args.event_pairs)
        dataset.save("pipeline_tmp_" + args.event_detector + "_" + args.event_pairs + "/" + document_id + ".pt")
    relations = predict_temporal_relations(dataset, relation_detection_method=relation_detection_method, args=args)
    for i in range(len(dataset.generated)):
        relations[i] = tuple(list(relations[i]) + [1 if dataset.generated[i].gold_relations else 0])
    return relations

def get_event_pairs_of_interest(relations):
    event_pairs = []
    for g in relations:
        event_pairs.append((g[0], g[2]))
    return list(event_pairs)

def run_pipeline(args):
    relation_detection_method = "separate"
    if args.event_pairs == "gold_only":
        f = open("pipeline_predictions_" + args.event_detector + "_" + args.relation_model + "_gold_pairs.txt", "a")
    else:
        f = open("pipeline_predictions_" + args.event_detector + "_" + args.relation_model + ".txt", "a")
    documents, sentences, relations, document_ids = get_sentences_from_test_set_i2b2()
    patient_id = 0
    i = 0
    print (len(documents))
    for document, sentences, true_relations, document_id in zip(documents, sentences, relations, document_ids):
        event_pairs = get_event_pairs_of_interest(true_relations)
        predicted_relations = analyze_document(document, patient_id, args, event_pairs, relation_detection_method=relation_detection_method, document_id=document_id)
        if relation_detection_method == "separate":
            predicted_relations = [
                [[int(e1s), int(e1e), e1t], relation, [int(e2s), int(e2e), e2t], int(p), int(gold), confidence]
                for (e1s, e1e, e1t), relation, (e2s, e2e, e2t), p, confidence, gold in predicted_relations
            ]
        else:
            predicted_relations = [
                [[int(e1s), int(e1e), e1t], relation, [int(e2s), int(e2e), e2t], int(gold), confidence]
                for (e1s, e1e, e1t), relation, (e2s, e2e, e2t), confidence, gold in predicted_relations
            ]
        print(predicted_relations)
        true_relations = [
            [[int(e1s), int(e1e), e1t], relation, [int(e2s), int(e2e), e2t]]
            for (e1s, e1e, e1t), relation, (e2s, e2e, e2t) in true_relations
        ]
        # print(true_relations)
        f.write(json.dumps(predicted_relations) + "|" + json.dumps(true_relations) + "|" + document + "|" + document_id + "\n")
        f.flush()
        # save_to_common_graph(predicted_relations, patient_id)
        patient_id += 1
    f.flush()
    f.close()

if __name__ == '__main__':
    run_pipeline()
    # analyze_document("He came to the hospital for a checkup on a mole that appeared two weeks ago.", 0)