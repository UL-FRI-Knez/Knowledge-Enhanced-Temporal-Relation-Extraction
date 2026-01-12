import pandas as pd
import spacy
from tdc.resource import PrimeKG
from scispacy.linking import EntityLinker
import torch
from torch_geometric.data import Data

from custom_datasets.common import lock
from graph_building import node_embeddings
from graph_building.node_embeddings import sentence_embedding

primeKG_data = None
umls_to_mondo = None


def get_PrimeKG():
    global primeKG_data, umls_to_mondo
    if primeKG_data is None:
        primeKG_data = PrimeKG(path='./graph_building/PrimeKG/data')
        primeKG_data.to_nx()

    # nodes = pd.read_csv('./PrimeKG/github data/nodes.csv')
    # umls_codes = pd.read_csv('data/umls/umls.csv')

    if umls_to_mondo is None:
        umls_to_mondo = pd.read_csv('./graph_building/PrimeKG/data/vocab/umls_mondo.csv')
    return primeKG_data, umls_to_mondo


nodes = None


def get_nodes():
    global nodes
    if nodes is None:
        nodes = pd.read_csv('./graph_building/PrimeKG/github data/nodes.csv')
    return nodes


nlp = None


def link_to_umls(entity):
    global nlp, umls_to_mondo
    with lock:
        if nlp is None:
            nlp = spacy.load("en_core_sci_sm")
            nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    _, umls_to_mondo = get_PrimeKG()
    entities = nlp(entity)
    if len(entities.ents) > 0:
        for ent in entities.ents:
            for cuid, conf in ent._.kb_ents:
                cuid = str(cuid)
                if cuid in list(umls_to_mondo['umls_id']):
                    if len(umls_to_mondo.query('umls_id == "'+cuid+'"')['mondo_id']) > 0:
                        return cuid, int(umls_to_mondo.query('umls_id == "'+cuid+'"')['mondo_id'].iloc[0])
    return None, None


def linked_umls(cuid):
    if cuid in list(umls_to_mondo['umls_id']):
        if len(umls_to_mondo.query('umls_id == "' + cuid + '"')['mondo_id']) > 0:
            return cuid, int(umls_to_mondo.query('umls_id == "' + cuid + '"')['mondo_id'].iloc[0])
    return cuid, None


disease_feature = None
def get_node_details(mondo, entity_name):
    global disease_feature
    PrimeKG, _ = get_PrimeKG()
    if disease_feature is None:
        disease_feature = PrimeKG.get_features(feature_type='disease')
    nodes = get_nodes()
    if isinstance(mondo, int) or mondo.isnumeric():
        features_disease = disease_feature.query('mondo_id == ' + str(mondo) + '')
    else:
        print("Warning: invalid mondo_id")
        features_disease = disease_feature.query('mondo_id == "' + str(mondo) + '"')
    definitions_and_descriptions = []
    for feature in features_disease.iloc:
        definitions_and_descriptions.append(feature['umls_description'])
        definitions_and_descriptions.append(feature['mondo_definition'])
    definitions_and_descriptions = list(dict.fromkeys(definitions_and_descriptions))
    names = []
    basic_data = nodes.query(
        'node_id == "' + str(mondo) + '"' + ' & (node_type == "MONDO_grouped" | node_type == "MONDO")')
    for feature in basic_data.iloc:
        names.append(feature['node_name'])
    return {"definitions": definitions_and_descriptions, "name": names[0] if len(names) > 0 else str(entity_name)}


def get_node_embedding(concept_description):
    if type(concept_description) == str:
        return node_embeddings.sentence_embedding(concept_description)
    definitions = concept_description['definitions']
    name = concept_description['name']
    combined_description = str(name)
    for definition in definitions:
        if type(definition) == str:
            combined_description = combined_description + '\n' + definition
    return node_embeddings.sentence_embedding(combined_description)

def get_link_embedding(relation):
    return node_embeddings.sentence_embedding(relation)


def get_subgraph(entity, entity_name):
    cuid, mondo = linked_umls(entity)

    if mondo is not None:
        primeKG, _ = get_PrimeKG()
        links = primeKG.df.query('x_id == ' + str(mondo))
        concepts = {mondo: get_node_details(mondo, entity_name)}
        concept_index = [mondo]
        relations = [('self', mondo, mondo)]
        for link in links.iloc:
            relations.append((link['relation'], link['x_id'], link['y_id']))
            target = link['y_id']
            if target not in concepts:
                concepts[target] = get_node_details(target, target)
                concept_index.append(target)
            pass
    else:
        concepts = {entity: {"definitions": [], "name": entity_name}}
        concept_index = [entity]
        relations = [('self', entity, entity)]


    # display_graph(concepts, relations)

    # convert to torch geometric
    x = []
    edge_index = [[], []]
    edge_features = []
    for c in concept_index:
        x.append(get_node_embedding(concepts[c]))
    for r in relations:
        edge_features.append(get_link_embedding(r[0]))
        edge_index[0].append(concept_index.index(r[1]))
        edge_index[1].append(concept_index.index(r[2]))

    data = Data(x=torch.cat(x, dim=0), edge_index=torch.Tensor(edge_index), edge_attr=torch.cat(edge_features, dim=0), term_index=0)
    return data


def display_graph(concepts, links):
    for link in links:
        c1 = concepts[link[1]]['name']
        c2 = concepts[link[2]]['name']
        link = link[0]
        print(c1, "--", link, "->", c2)

if __name__ == '__main__':
    print(get_subgraph("C0349644"))
    pass
