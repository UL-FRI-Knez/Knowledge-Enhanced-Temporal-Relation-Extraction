import datetime
import random
import threading

import gensim
import gensim.downloader
import numpy as np
import spacy
from scispacy.linking import EntityLinker
import torch
import torch_geometric
from torch_geometric.data import Data

from custom_datasets.common import compute_transitive_relation, custom_map, Configuration, lock
from custom_datasets import combining_data
import torch.nn.functional as F

from custom_datasets.combining_data import read_i2b2
from graph_building import node_embeddings

index_to_label = ['BEFORE', 'AFTER', 'OVERLAP', 'BEGINS-ON', 'CONTAINED-BY', 'CONTAINS', 'ENDS-ON', 'CONTINUES', 'TERMINATES', 'INITIATES', 'REINITIATES']
labels = {'BEFORE': 0,
          'AFTER': 1,
          'OVERLAP': 2,
          'BEGINS-ON': 3,
          'CONTAINED-BY': 4,
          'CONTAINS': 5,
          'ENDS-ON': 6,
          'CONTINUES': 7,
          'TERMINATES': 8,
          'INITIATES': 9,
          'REINITIATES': 10,
          }

nlp = None
glove_vectors = None

def link_entity_to_umls(entity):
    global nlp
    with lock:
        if nlp is None:
            nlp = spacy.load("en_core_sci_sm")
            nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    entities = nlp(entity)
    if len(entities.ents) > 0:
        linked_entities = entities.ents[0]._.kb_ents
        if len(linked_entities) > 0:
            return linked_entities[0][0]
    return entity

def filter_knowledge_graph_document(grpah, document):
    return list(filter(lambda edge: document == edge[3], grpah))

def filter_knowledge_graph(grpah, nodes):
    return list(filter(lambda edge: not (edge[0] in nodes and edge[2] in nodes), grpah))

def add_inverse_relations(graph):
    set_of_relations = set([tuple(r) for r in graph])
    new_relations = []
    for relation in graph:
        inverse_relation = relation.copy()
        inverse_relation[0], inverse_relation[2] = inverse_relation[2], inverse_relation[0]
        inverse_relation[1] = combining_data.relation_inverse(inverse_relation[1])
        if len(inverse_relation) > 4:
            conf_relation = list(inverse_relation[4])
            conf_relation[0], conf_relation[1] = conf_relation[1], conf_relation[0]
            inverse_relation[4] = tuple(conf_relation)
        if len(inverse_relation) > 6:
            inverse_relation[5], inverse_relation[6] = inverse_relation[6], inverse_relation[5]
        if tuple(inverse_relation) not in set_of_relations:
            new_relations.append(inverse_relation)
            set_of_relations.add(tuple(inverse_relation))
    return graph + new_relations


def add_transitive_relations(graph, only_additional=False):
    set_of_relations = set([tuple(r) for r in graph])
    new_relations = []
    for r1 in graph:
        for r2 in graph:
            if r1[3] == r2[3] and r1[2] == r2[0]:
                # relations are neighbours
                new_relation = r1.copy()
                relation_text = compute_transitive_relation(r1[1], r2[1])
                new_relation[1] = relation_text
                new_relation[2] = r2[2]
                if tuple(new_relation) not in set_of_relations:
                    set_of_relations.add(tuple(new_relation))
                    new_relations.append(new_relation)
    if only_additional:
        return new_relations
    else:
        return graph + new_relations

def get_relations(graph, nodes):
    relations = []
    for g in graph:
        if g[0] in nodes and g[2] in nodes:
            relations.append(g)
    return relations


def convert_df_row(row):
    text = row['text']
    y = row['class']
    event1_start = row['event1_start']
    event2_start = row['event2_start']
    event1_end = row['event1_end']
    event2_end = row['event2_end']
    document_id = row['document_id']
    text, event1_start, event1_end, event2_start, event2_end = add_event_tokens(text, event1_start, event1_end,
                                                                                event2_start, event2_end)

    if "<" in text[event1_start:event1_end] or "<" in text[event2_start:event2_end]:
        # print("opozorilo")
        pass
    return text, event1_start, event1_end, event2_start, event2_end, \
           torch.tensor(labels[y]) if y else None, document_id

def add_event_tokens(text, event1_start, event1_end, event2_start, event2_end):
    tag_start1, tag_start2, tag_end1, tag_end2 = "<e1>", "<e2>", "</e1>", "</e2>"
    # tag_start1, tag_start2, tag_end1, tag_end2 = "<e>", "<e>", "</e>", "</e>"
    text = text[:event1_start] + tag_start1 + text[event1_start:]
    if event1_end >= event1_start:
        event1_end += len(tag_start1)
    if event2_start >= event1_start:
        event2_start += len(tag_start1)
    if event2_end >= event1_start:
        event2_end += len(tag_start1)
    if event1_start >= event1_start:
        event1_start += len(tag_start1)

    text = text[:event1_end] + tag_end1 + text[event1_end:]
    if event1_start > event1_end:
        event1_start += len(tag_end1)
    if event2_start > event1_end:
        event2_start += len(tag_end1)
    if event2_end > event1_end:
        event2_end += len(tag_end1)

    if max(event1_start, event2_start) < min(event1_end, event2_end):
        return text, event1_start, event1_end, event1_start, event1_end

    text = text[:event2_start] + tag_start2 + text[event2_start:]
    if event1_start >= event2_start:
        event1_start += len(tag_start2)
    if event1_end >= event2_start:
        event1_end += len(tag_start2)
    if event2_end >= event2_start:
        event2_end += len(tag_start2)
    if event2_start >= event2_start:
        event2_start += len(tag_start2)

    text = text[:event2_end] + tag_end2 + text[event2_end:]
    if event1_start >= event2_end:
        event1_start += len(tag_end2)
    if event1_end >= event2_end:
        event1_end += len(tag_end2)
    if event2_start >= event2_end:
        event2_start += len(tag_end2)
    return text, event1_start, event1_end, event2_start, event2_end

def get_neighbourhood(graph, initial_entities, steps=3):
    entities = set()
    for e in initial_entities:
        entities.add(e)
    for k in range(steps):
        new_entities = set()
        for relation in graph:
            if relation[0] in entities:
                new_entities.add(relation[2])
            if relation[2] in entities:
                new_entities.add(relation[0])
        entities = entities.union(new_entities)
    return list(entities)

def get_path(graph, entity1_id, entity2_id, max_steps=5, return_multiple=False):
    visited = set()
    visited.add(entity1_id)
    source = {}
    queue = [(entity1_id, 0)]
    paths = []
    shortest_path = max_steps
    while len(queue) > 0:
        q = queue.pop(0)
        if q[1] > max_steps:
            if not return_multiple:
                return None
            else:
                return paths
        if q[1] > shortest_path:
            break
        f = q[0]
        for triplet in graph:
            if triplet[0] == f and triplet[2] not in visited:
                visited.add(triplet[2])
                source[triplet[2]] = f
                queue.append((triplet[2], q[1] + 1))
                if triplet[2] == entity2_id:
                    # we found target
                    node = entity2_id
                    path = [node]
                    while node in source:
                        node = source[node]
                        path.insert(0, node)
                    if not return_multiple:
                        return path
                    paths.append(path)
                    shortest_path = len(path)
    return paths
def get_multiword_word2vec(entity):
    global glove_vectors
    if glove_vectors is None:
        glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')
    vectors = []
    for word in entity.split():
        word = word.lower()
        if word in glove_vectors.key_to_index:
            vectors.append(glove_vectors.vectors[glove_vectors.key_to_index[word]])
    if len(vectors) == 0:
        vectors = [np.random.rand(50)] # TODO you might be able to find better embeddings than random ones
    return np.array(vectors).mean(0)

def get_embedding_for_entity(entity):
    global nlp, glove_vectors
    with lock:
        if nlp is None:
            nlp = spacy.load("en_core_sci_sm")
            nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        if glove_vectors is None:
            glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')
    linker = nlp.get_pipe("scispacy_linker")
    if entity in linker.kb.cui_to_entity:
        canonical_name = linker.kb.cui_to_entity[entity].canonical_name
        definition = linker.kb.cui_to_entity[entity].definition
        # use bert embeddings instead of word2vec
        vector = node_embeddings.sentence_embedding(canonical_name).detach()
        # vector = get_multiword_word2vec(canonical_name)
    else:
        vector = node_embeddings.sentence_embedding(entity).detach()
        # vector = get_multiword_word2vec(entity)
    return vector

def generate_graph_for_gnn(graph, entity1, entity2, y, text_features=None, use_entire_graph=True, neighbourhood=False):
    nodes = []
    if use_entire_graph:
        nodes = [r[0] for r in graph] + [r[2] for r in graph]
    else:
        for path in get_path(graph, entity1, entity2, return_multiple=True):
            nodes += path
    nodes = list(set(nodes))
    if nodes is None or len(nodes) == 0:
        nodes = [entity1, entity2]
    if entity1 not in nodes:
        nodes.append(entity1)
    if entity2 not in nodes:
        nodes.append(entity2)

    if neighbourhood:
        nodes = get_neighbourhood(graph, nodes, steps=1)
    node_to_index = {}
    for i, node in enumerate(nodes):
        node_to_index[node] = i
    relations = get_relations(graph, nodes)
    edge_index = []
    edge_type = []
    edge_attr = []
    text_relations = []
    for r in relations:
        text_relations.append((r[5], r[1], r[6]))
        edge_index.append([node_to_index[r[0]], node_to_index[r[2]]])
        edge_type.append(labels[r[1]])
        if len(r) > 4:
            edge_attr.append(r[4])
        else:
            edge_attr.append(tuple(F.one_hot(torch.tensor(labels[r[1]]), 3).tolist()))
    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((0, 0))
    embeddings = [get_embedding_for_entity(n) for n in nodes]
    # x = torch.ones(len(nodes), 50)
    x = torch.cat(embeddings, 0)
    edge_type = torch.tensor(edge_type)
    index1 = node_to_index[entity1]
    index2 = node_to_index[entity2]
    rule_based_prediction = rule_based_model(graph, entity1, entity2)

    # Text features
    if text_features is not None:
        text, event1_start, event1_end, event2_start, event2_end, _, document_id = text_features
        data = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=torch.tensor([y]) if y is not None else None,
                    event1_index=index1, event2_index=index2, rule_based_prediction=rule_based_prediction,
                    text=text, event1_start=event1_start, event1_end=event1_end, event2_start=event2_start, event2_end=event2_end, document_id=document_id,
                    edge_attr=torch.tensor(edge_attr))
    else:
        data = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=torch.tensor([y]) if y is not None else None,
                    event1_index=index1, event2_index=index2, rule_based_prediction=rule_based_prediction)
    return data, text_relations

def path_to_relations(graph, path):
    relation_sequence = []
    for i in range(len(path) - 1):
        e1 = path[i]
        e2 = path[i + 1]
        for r in graph:
            if r[0] == e1 and r[2] == e2:
                relation_sequence.append(r[1])
                break
    return relation_sequence

def rule_based_model(graph, event1, event2):
    path = get_path(graph, event1, event2)
    if path is None:
        return 2
    relation_sequence = path_to_relations(graph, path)
    relation = 0
    for r in relation_sequence:
        if r == "BEFORE":
            relation -= 1
        if r == "AFTER":
            relation += 1
    if relation > 0:
        return 1
    elif relation < 0:
        return 0
    else:
        return 2

def create_graph(iteration, **kwargs):
    row, graph, configuration = iteration
    if type(row) is tuple or type(row) is list:
        row = row[1]
    document_id = row["document_id"]
    event1 = link_entity_to_umls(row["event1_text"])
    event2 = link_entity_to_umls(row["event2_text"])
    if configuration.no_document_filtering:
        active_graph = graph
    else:
        active_graph = filter_knowledge_graph_document(graph, document_id)
    if configuration.add_inverse_relations_to_graph:
        active_graph = add_inverse_relations(active_graph)
    if configuration.add_transitive_relations_to_graph:
        transitive_relations = add_transitive_relations(active_graph, only_additional=True)
        active_graph += transitive_relations

    # relations from other files
    if not configuration.no_document_filtering and configuration.use_relations_from_other_documents:
        relations_from_other_files = list(filter(lambda r: r[3] != row["document_id"], get_relations(graph, [event1, event2])))
        if configuration.add_inverse_relations_to_graph:
            relations_from_other_files = add_inverse_relations(relations_from_other_files)

    # remove random relations to simulate wrongly labeled relations
    if configuration.simulated_realistic_graph['use_simulated_realistic_graph']:
        active_graph = random.sample(active_graph, int(len(active_graph) * configuration.simulated_realistic_graph['portion_of_relations_to_keep']))

    if not configuration.no_document_filtering and configuration.use_relations_from_other_documents:
        active_graph = active_graph + relations_from_other_files

    # text features
    text_features = convert_df_row(row)

    if configuration.remove_target_relation:
        active_graph = filter_knowledge_graph(active_graph, [event1, event2])
    subgraph, text_triplets = generate_graph_for_gnn(active_graph, event1, event2,
                                      labels[row["class"]] if row["class"] else None, text_features, use_entire_graph=configuration.use_entire_graph)
    if len(subgraph.x) == 0:
        return None
    if "return_text_triplets" in kwargs and kwargs["return_text_triplets"]:
        return subgraph, text_triplets
    else:
        return subgraph

class KnowledgeGraphDataset(torch.utils.data.Dataset):
    def __init__(self, graph, df, number_of_classes=3, simulate_wrong_relations=False, configuration=Configuration()):
        self.graphs = []
        self.labels = []

        global nlp, glove_vectors
        with lock:
            if nlp is None:
                nlp = spacy.load("en_core_sci_sm")
                nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
                if glove_vectors is None:
                    glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')

        # with Pool(8) as pool:
        #     self.graphs = pool.map(create_graph, [(row, graph) for row in df.iterrows()])
        # self.graphs = list(map(create_graph, [(row, graph, configuration) for row in df.iterrows()]))
        self.graphs = list(custom_map(create_graph, [(row, graph, configuration) for row in df.iterrows()]))
        self.graphs = [g for g in self.graphs if g is not None]
        self.num_node_features = self.graphs[0].x.shape[1]
        self.labels = [x.y for x in self.graphs]

        self.num_classes = number_of_classes
        random.Random(1).shuffle(self.graphs)

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.graphs[idx].y

        item = {'graph': self.graphs[idx]}
        item['labels'] = torch.tensor(self.graphs[idx].y)
        return item

def prepare_local_graph_dataset(full_text_df, configuration):
    test_df = combining_data.add_inverse_relations(full_text_df)
    test_df = combining_data.add_transitive_relations(test_df)
    test_df = combining_data.window_for_entity_bert(test_df, window_size=60, normalize_event_order=True)
    test_df = test_df.drop_duplicates(subset=['text', 'class', 'event1_start', 'event2_start'], keep='last').reset_index()
    test_dataset = KnowledgeGraphDataset([], test_df, configuration=configuration)
    return test_dataset

def construct_graph_from_text_only(full_text_df, configuration, dataset_type=""):
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_df = combining_data.add_inverse_relations(full_text_df)
    test_df = combining_data.add_transitive_relations(test_df)
    test_df = combining_data.window_for_entity_bert(test_df, window_size=60, normalize_event_order=True)
    test_df = test_df.drop_duplicates(subset=['text', 'class', 'event1_start', 'event2_start'], keep='last').reset_index(drop=True)
    # TODO drop self loops
    test_dataset = KnowledgeGraphDataset([], test_df, configuration=configuration)
    dataLoader_test = torch_geometric.loader.DataLoader(test_dataset, batch_size=batch_size)

    text_model = torch.load("graph_building/local_graph/pretrained_models/Transitive_relation_extraction.pt", map_location=torch.device(device))
    # text_model = torch.load("checkpoints/EntityBert_relation_extraction.pt", map_location=torch.device(device))
    in_memory_graph = []

    text_model.eval()
    text_model.training = False
    labels = []
    predictions = []
    raw_predictions = None
    correct = 0
    n = 0
    n_all = 0
    for batch in dataLoader_test:
        graph, _ = batch
        y = graph.y
        graph.to(device)
        y = y.to(device)
        res = text_model(graph)
        res = torch.softmax(res, dim=1)
        if raw_predictions is None:
            raw_predictions = res
        else:
            raw_predictions = torch.cat((raw_predictions, res), 0)
        pred = res.argmax(dim=1)

        lab = y.tolist()
        labels += lab
        pred = pred.tolist()
        predictions += pred
        for i in range(len(pred)):
            n_all += 1
            if configuration.realistic_graph['remove_wrong_edges'] and pred[i] != lab[i]:
                continue
            if configuration.use_realistic_graph:
                p = pred[i]
                raw_pred = tuple(res[i].tolist())
            else:
                p = lab[i]
                raw_pred = tuple(F.one_hot(torch.tensor(lab[i]), 3).tolist())
            if configuration.realistic_graph['use_threshold_confidence'] and \
                    raw_pred[p] < configuration.realistic_graph['threshold_confidence']:
                continue
            if p == lab[i]:
                correct += 1
            n += 1
            raw_event1 = graph.text[i][graph.event1_start[i]:graph.event1_end[i]]
            event1 = link_entity_to_umls(raw_event1)
            raw_event2 = graph.text[i][graph.event2_start[i]:graph.event2_end[i]]
            event2 = link_entity_to_umls(raw_event2)
            document_id = graph.document_id[i]
            relation = index_to_label[p]
            in_memory_graph.append([event1, relation, event2, document_id, raw_pred, raw_event1, raw_event2])
    print("Accuracy:", correct / n)
    print("Number of relations:", n / n_all)
    file1 = open("construct graph from text only.log", "a")  # append mode
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file1.write("Data type=" + dataset_type + ", " + "Date=" + str(now) + ", " + str(configuration) + ", Accuracy=" + str(correct/n) + ", Number of relations=" + str(n / n_all) + " \n")
    file1.close()
    in_memory_graph = add_inverse_relations(in_memory_graph)
    return in_memory_graph

def precompute_local_knowledge_graph():
    configuration = Configuration()
    configuration.add_inverse_relations_to_graph = True
    configuration.remove_target_relation = False
    configuration.use_realistic_graph = True
    df = read_i2b2(full_text=True, use_test_files=False, include_rows_without_absolute=True)
    df = df[:54]
    in_memory_kg = construct_graph_from_text_only(df, configuration, dataset_type="train")
    torch.save(in_memory_kg, "computed_kg.pt")

if __name__ == '__main__':
    configuration = Configuration()
    configuration.add_inverse_relations_to_graph = True
    configuration.remove_target_relation = False
    configuration.use_realistic_graph = True
    df = read_i2b2(full_text=True, use_test_files=False, include_rows_without_absolute=True)
    in_memory_kg = construct_graph_from_text_only(df, configuration, dataset_type="train")
    torch.save(in_memory_kg, "computed_kg.pt")