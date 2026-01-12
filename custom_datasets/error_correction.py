import re

import torch
import torch.nn.functional as F

import date2vec.date_embedding
from custom_datasets import combining_data
from graph_building.node_embeddings import sentence_embedding


def generate_edge_embedding(edge_type, edge_type_index, edge_tensor, word_embedding_size=768):
    edge_type_tensor = F.one_hot(torch.tensor(edge_type_index), 4)

    if edge_type == "temporal_relation":
        return torch.cat((edge_type_tensor, edge_tensor[:3], torch.zeros(word_embedding_size)))
    elif edge_type == "date":
        return torch.cat((edge_type_tensor, torch.zeros(3), edge_tensor))
    elif edge_type == "general_relation":
        return torch.cat((edge_type_tensor, torch.zeros(3), edge_tensor))
    elif edge_type == "document_part":
        return torch.cat((edge_type_tensor, torch.zeros(3 + word_embedding_size)))
    raise Exception("Invalid edge type: " + edge_type)

def update_pregenerated_graph(graph, row):
    # recognise where relations came from
    edges_local_graph_start = 0
    edges_primekg_start = 0
    for edge_index in range(len(graph.edge_attr)):
        graph.edge_attr[edge_index]
        if graph.edge_attr[edge_index][3:].abs().sum() == 0:
            edges_primekg_start = edge_index
        elif edges_primekg_start == 0:
            edges_local_graph_start = edge_index
    edges_local_graph_start += 1
    edges_primekg_start += 1

    nodes_local_graph_start = 0
    nodes_primekg_start = 0
    for i in range(0, edges_local_graph_start):
        nodes_local_graph_start = max(nodes_local_graph_start, int(graph.edge_index[0][i]))
        nodes_local_graph_start = max(nodes_local_graph_start, int(graph.edge_index[1][i]))
    nodes_local_graph_start += 1
    for i in range(edges_local_graph_start, edges_primekg_start):
        nodes_primekg_start = max(nodes_primekg_start, int(graph.edge_index[0][i]))
        nodes_primekg_start = max(nodes_primekg_start, int(graph.edge_index[1][i]))
    nodes_primekg_start += 1
    edges_original_end = len(graph.edge_attr)
    nodes_original_end = len(graph.x)

    # add additional features
    edge_features = []
    edge_types = []
    for i in range(len(graph.edge_attr)):
        if i < edges_local_graph_start:
            # llm graph
            edge_type = 'general_relation'
        elif i >= edges_local_graph_start and i < edges_primekg_start:
            # local graph
            edge_type = 'temporal_relation'
        elif i >= edges_primekg_start:
            # primekg graph
            edge_type = 'general_relation'

        edge_type_index = ["date", "temporal_relation", "general_relation", "document_part"].index(edge_type)
        edge_features.append(generate_edge_embedding(edge_type, edge_type_index, graph.edge_attr[i]))
        edge_types.append(edge_type_index)

    # add document node connected to all nodes from the document
    graph.x = torch.cat((graph.x, sentence_embedding("Document")))
    document_node_index = len(graph.x) - 1
    new_edges = [[],[]]
    for i in range(nodes_local_graph_start, nodes_primekg_start):
        new_edges[0].append(document_node_index)
        new_edges[1].append(i)
        edge_features.append(generate_edge_embedding('document_part', 3, None))
        edge_types.append(3)


    # add new nodes
    discharge, admission = get_more_information_from_graph(row.text, row.event1_start, row.event2_start, row.event1_end, row.event2_end)
    graph.x = torch.cat((graph.x, sentence_embedding("Admission"), sentence_embedding("Discharge")))
    new_edges[0].append(document_node_index)
    new_edges[1].append(len(graph.x)-2)
    new_edges[0].append(document_node_index)
    new_edges[1].append(len(graph.x)-1)
    edge_features.append(generate_edge_embedding('date', 0, date2vec.date_embedding.compute_date_embedding(*admission)))
    edge_types.append(0)
    edge_features.append(generate_edge_embedding('date', 0, date2vec.date_embedding.compute_date_embedding(*discharge)))
    edge_types.append(0)

    graph.edge_index = torch.cat((graph.edge_index, torch.tensor(new_edges)), dim=1)

    edge_attr = torch.cat([x.reshape(-1, 1) for x in edge_features], dim=1).T
    graph.edge_attr = edge_attr
    graph.edge_type = edge_types
    return graph

def fix_precomputed_dataset(dataset):
    for i in range(len(dataset.generated)):
        dataset.generated[i] = update_pregenerated_graph(dataset.generated[i])
    return dataset

i2b2_dataset = None
def get_more_information_from_graph(text, event1_start, event2_start, event1_end, event2_end):
    # TODO add support for datasets other than i2b2
    global i2b2_dataset
    if i2b2_dataset is None:
        i2b2_dataset = combining_data.read_i2b2(full_text=True, use_test_files=False, include_rows_without_absolute=True)

    row = i2b2_dataset[(i2b2_dataset["text"].str.contains(re.escape(text))) &
                       (i2b2_dataset["event1_start"] == event1_start) &
                       (i2b2_dataset["event1_end"] == event1_end) &
                       (i2b2_dataset["event2_start"] == event2_start) &
                       (i2b2_dataset["event2_end"] == event2_end)]
    admission = [0,0,0,0,0,0]
    dct = [0,0,0,0,0,0]
    def convert_time_string_to_tuple(time_strign):
        x = re.search("(\d{4})-(\d{1,2})-(\d{1,2})", time_strign)
        return [x[1], x[2], x[3], 0, 0, 0]
    if len(row) >= 1:
        for time in row["additional_document_info"].iloc[0]["times"]:
            if time[0] == "ADMISSION":
                admission = convert_time_string_to_tuple(time[1])
            if time[0] == "DISCHARGE":
                dct = convert_time_string_to_tuple(time[1])
    return dct, admission


if __name__ == '__main__':
    graph = torch.load("testni_graf.pt")
    # get_more_information_from_graph(graph.text, graph.event1_start, graph.event2_start, graph.event1_end, graph.event2_end)
    print(update_pregenerated_graph(graph))