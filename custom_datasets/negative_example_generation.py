import random

from torch_geometric.data import Data
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import torch
import os, psutil
import gc

results_file = "evaluation_results_no_relation/results-test-battery.txt"

def add_null_relations(dataset):
    with open(results_file, "a") as myfile:
        myfile.write("\nAdding null relations\n")
        myfile.flush()
    print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    gc.collect()
    print("running add_null_relations")
    null_relation_index = dataset.number_of_relations()

    dataframe = dataset.df
    documents = set(dataframe["document_id"])
    train_documents = []

    for doc in documents:
        train_documents.append(dataframe[dataframe["document_id"] == doc].reset_index(drop=True))


    nlp = English()
    tokenizer = Tokenizer(nlp.vocab)

    def get_token_for_char(tokens, char_idx):
        for i, token in enumerate(tokens):
            if char_idx > token.idx:
                continue
            if char_idx == token.idx:
                return i, token
            if char_idx < token.idx:
                return i - 1, tokens[i - 1]
        return len(tokens) - 1, tokens[len(tokens) - 1]

    def window_row_for_bert(text, e1, e2, window_size=60, add_event_markers=True):
        tokens = tokenizer(text)
        start = min(e1[0], e2[0])
        end = max(e1[1], e2[1])
        start_token, _ = get_token_for_char(tokens, start)
        end_token, _ = get_token_for_char(tokens, end)
        if end_token - start_token > window_size:
            return None
        start_token -= (window_size - (end_token - start_token)) // 2
        end_token += (window_size - (end_token - start_token)) // 2
        end_token += max(0, -start_token)
        start_token = max(0, start_token)
        end_token = min(end_token, len(tokens) - 1)
        start = tokens[start_token].idx
        end = tokens[end_token].idx + len(tokens[end_token])
        text = text[start:end]
        e1s = e1[0] - start
        e1e = e1[1] - start
        e2s = e2[0] - start
        e2e = e2[1] - start
        if add_event_markers:
            text = text[:e1s] + "<e1>" + text[e1s:e1e] + "</e1>" + text[e1e:]
            if e2s > e1s:
                e2s += 4
            if e2e > e1s:
                e2e += 4
            if e2s > e1e:
                e2s += 5
            if e2e > e1e:
                e2e += 5
            e1s += 4
            e1e += 4

            text = text[:e2s] + "<e2>" + text[e2s:e2e] + "</e2>" + text[e2e:]
            if e1s > e2s:
                e1s += 4
            if e1e > e2s:
                e1e += 4
            if e1s > e2e:
                e1s += 5
            if e1e > e2e:
                e1e += 5
            e2s += 4
            e2e += 4
        e1 = (e1s, e1e, e1[2])
        e2 = (e2s, e2e, e2[2])
        return text, e1, e2

    def convert_to_graph(df):
        graph = []
        text = ""
        events = set()
        for row in df.iloc:
            text = row["text"]
            graph.append(((row["event1_start"], row["event1_end"], row["event1_text"]), row["class"],
                          (row["event2_start"], row["event2_end"], row["event2_text"])))
            events.add((row["event1_start"], row["event1_end"], row["event1_text"]))
            events.add((row["event2_start"], row["event2_end"], row["event2_text"]))
        return graph, text, events

    transitivity = {
        ("OVERLAP", "OVERLAP"): "OVERLAP",
        ("BEFORE", "OVERLAP"): "BEFORE",
        ("BEFORE", "BEFORE"): "BEFORE",
        ("OVERLAP", "BEFORE"): "BEFORE",
        ("AFTER", "OVERLAP"): "AFTER",
        ("OVERLAP", "AFTER"): "AFTER",
        ("AFTER", "AFTER"): "AFTER",
        ("AFTER", "BEFORE"): "OVERLAP",
        ("BEFORE", "AFTER"): "OVERLAP"
    }

    inverse = {
        "BEFORE": "AFTER",
        "AFTER": "BEFORE",
        "OVERLAP": "OVERLAP"
    }

    def find_relation(graph, event1, event2, expected_relation):
        for ind, r in enumerate(graph):
            if r[0] == event1 and r[2] == event2 and r[1] == expected_relation:
                return ind, r
        for ind, r in enumerate(graph):
            if r[0] == event1 and r[2] == event2:
                return ind, r
        return -1, None

    def does_relation_exist(graph, event1, event2):
        ind, relation = find_relation(graph, event1, event2, None)
        return ind >= 0

    def add_transitive(graph):
        graph_len = len(graph)
        for i in range(graph_len):
            for j in range(i + 1, graph_len):
                if graph[i][2] == graph[j][0]:
                    # matching relations
                    event1 = graph[i][0]
                    event2 = graph[j][2]
                    if (graph[i][1], graph[j][1]) in transitivity and not does_relation_exist(graph, event1, event2):
                        relation = transitivity[(graph[i][1], graph[j][1])]
                        new_relation = (event1, relation, event2)
                        graph.append(new_relation)
        return graph

    def add_inverse(graph):
        graph_len = len(graph)
        for i in range(graph_len):
            event1 = graph[i][2]
            event2 = graph[i][0]
            if not does_relation_exist(graph, event1, event2):
                relation = inverse[graph[i][1]]
                new_relation = (event1, relation, event2)
                graph.append(new_relation)
        return graph

    def graph_closure(graph):
        graph = add_inverse(graph)
        graph = add_transitive(graph)
        return graph

    graphs = []
    for i in range(len(train_documents)):
        graphs.append(convert_to_graph(train_documents[i]))

    # compute closures

    # for i in range(len(train_documents)):
    #     graphs[i] = (graph_closure(graphs[i][0]), graphs[i][1], graphs[i][2])

    # generate null examples
    initial_generated_size = len(dataset.generated)
    for graph, text, events in graphs:
        # add random examples
        events = list(events)
        events1 = random.sample(events, min(10, len(events)))
        events2 = random.sample(events, min(10, len(events)))
        for i in range(len(events1)):
            for j in range(len(events2)):
                e1 = events1[i]
                e2 = events2[j]
                window = window_row_for_bert(text, e1, e2)
                if window == None:
                    continue
                exists = does_relation_exist(graph, e1, e2)
                if exists:
                    # skip positive examples, as they were added already
                    continue
                windowed_text, e1, e2 = window

                # find the graphs for the events
                e1_x, e2_x = None, None
                e1_edge_index, e2_edge_index = None, None
                e1_edge_attr, e2_edge_attr = None, None
                e1_index, e2_index = None, None
                for j in range(initial_generated_size):
                    generated_graph = dataset.generated[j]
                    e1_text = generated_graph.text[generated_graph.event1_start:generated_graph.event1_end]
                    e2_text = generated_graph.text[generated_graph.event2_start:generated_graph.event2_end]
                    if e1_text == e1[2]:
                        e1_x = generated_graph.x
                        e1_edge_index = generated_graph.edge_index
                        e1_edge_attr = generated_graph.edge_attr
                        e1_index = generated_graph.event1_index
                    if e2_text == e1[2]:
                        e1_x = generated_graph.x
                        e1_edge_index = generated_graph.edge_index
                        e1_edge_attr = generated_graph.edge_attr
                        e1_index = generated_graph.event2_index
                    if e1_text == e2[2]:
                        e2_x = generated_graph.x
                        e2_edge_index = generated_graph.edge_index
                        e2_edge_attr = generated_graph.edge_attr
                        e2_index = generated_graph.event1_index
                    if e2_text == e2[2]:
                        e2_x = generated_graph.x
                        e2_edge_index = generated_graph.edge_index
                        e2_edge_attr = generated_graph.edge_attr
                        e2_index = generated_graph.event2_index

                if e1_x is not None and e2_x is not None:
                    data = Data(
                                x=torch.cat((e1_x, e2_x)),
                                y=torch.tensor([null_relation_index]),
                                edge_index=torch.cat((e1_edge_index, e2_edge_index + e1_x.size()[0]), dim=1),
                                edge_attr=torch.cat((e1_edge_attr, e2_edge_attr)),
                                event1_index=e1_index, event2_index=e2_index + e1_x.size()[0], text=windowed_text,
                                event1_start=e1[0],
                                event2_start=e2[0],
                                event1_end=e1[1],
                                event2_end=e2[1])
                    print("new null sample")
                    dataset.generated.append(data)
    return dataset