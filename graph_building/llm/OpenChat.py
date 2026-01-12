import json
import re
from difflib import SequenceMatcher

import torch
from graph_building.graph_construction import get_link_embedding, get_node_embedding
from torch_geometric.data import Data

from langchain_community.llms import Ollama
ollama = Ollama(base_url='http://localhost:11434',
model="openchat:7b", num_predict=200)

def request_open_Chat(prompt):
    return ollama.invoke(prompt)

def extract_data_in_brackets(input_string):
    pattern = r"\[(.*?)\]"
    matches = re.findall(pattern, input_string)
    return matches

def parse(response):
    # [[Broken arm, is an, orthopedic injury], [Broken arm, can cause, pain],
    # [Broken arm, may require, casting], [x-ray, used to diagnose, broken arm],
    # [physical therapy, helps with, recovery from broken arm], [fracture, is a
    # type of, broken arm], [doctor, treats, broken arm], [orthopedic surgeon,
    # specializes in treating, broken arm]]

    parsed_triplets = []
    response = re.sub("\]\s{0,3},?\s{0,3}\[", ";", response)
    response = re.sub("\n\d{0,4}\.?\s?", ";", response)
    response = response.replace("\"", "")
    response = response.replace("'", "")
    response = response.replace("[[", "")
    response = response.replace("]]", "")
    triplets = response.split(";")
    for t in triplets:
        parts = [p for p in t.split(",") if len(p.strip()) > 0]
        if len(parts) > 3:
            parts = [parts[0], " ".join(parts[1:-1]), parts[-1]]
        if len(parts) != 3:
            pass
        parts = [a.strip() for a in parts]
        if len(parts) != 3:
            print(parts)
            continue
        source, relation, target = parts
        if relation == "is a" or relation == "is an":
            relation = "is"
        source = source.lower()
        target = target.lower()
        parsed_triplets.append((source, relation, target))
    return parsed_triplets

def display_triplets(parsed_triplets):
    for source, relation, target in parsed_triplets:
        print(source, "--", relation, "->", target)
def get_kg_from_llm(term, category, response=None):
    prompt = generate_prompt_for_open_chat(term, category)
    if response is None:
        response = ollama.invoke(prompt)
    # Process the response to triples
    print(response)
    triples = parse(response)
    display_triplets(triples)
    data = convert_triplets_to_pyg(triples, term)
    return data, response

def similar(a, b):
    if a is None:
        a = ""
    if b is None:
        b = ""
    return SequenceMatcher(None, a, b).ratio()

def convert_triplets_to_pyg(triplets, term, start_index=0):
    term_idx = -1
    term_similarity = 0

    similarity_threshold = 0.8
    index = start_index
    nodes = {}

    nodes[term] = index
    index += 1

    edge_index = [[], []]
    edge_attr = []
    for triplet in triplets:
        source_idx = -1
        target_idx = -1
        # Find similar nodes
        for node in nodes:
            if similar(node, triplet[0]) > similarity_threshold:
                source_idx = nodes[node]
        if source_idx == -1:
            source_idx = index
            nodes[triplet[0]] = index
            index += 1
        for node in nodes:
            if similar(node, triplet[2]) > similarity_threshold:
                target_idx = nodes[node]
        if target_idx == -1:
            target_idx = index
            nodes[triplet[2]] = index
            index += 1
        edge_index[0].append(source_idx)
        edge_index[1].append(target_idx)
        edge_attr.append(get_link_embedding(triplet[1]))
        pass

    # convert nodes
    x = [0] * len(nodes)
    for idx, node in enumerate(nodes):
        x[nodes[node]] = get_node_embedding(node)
        if similar(node, term) > term_similarity:
            term_similarity = similar(node, term)
            term_idx = idx
    if len(edge_attr) == 0:
        print("No edge attributes found")
    if len(edge_attr) == 0:
        data = Data(x=torch.cat(x,0), edge_index=torch.Tensor(edge_index), edge_attr=torch.tensor([]), term_index=torch.Tensor([term_idx]))
    else:
        data = Data(x=torch.cat(x,0), edge_index=torch.Tensor(edge_index), edge_attr=torch.cat(edge_attr, 0), term_index=torch.Tensor([term_idx]))
    return data

def generate_prompt_for_open_chat(term, category):
    if category == "condition":
        example = \
        """
        Example:
        prompt: systemic lupus erythematosus
        updates: [[systemic lupus erythematosus, is an, autoimmune condition], [systemic
        lupus erythematosus, may cause, nephritis], [anti-nuclear antigen, is a test for,
        systemic lupus erythematosus], [systemic lupus erythematosus, is treated with,
        steroids], [methylprednisolone, is a, steroid]]
        """
    elif category == "procedure":
        example = \
        """
        Example:
        prompt: endoscopy
        updates: [[endoscopy, is a, medical procedure], [endoscopy, used for, diagnosis],
        [endoscopic biopsy, is a type of, endoscopy], [endoscopic biopsy, can detect,
        ulcers]]
        """
    elif category == "drug":
        example = \
        """
        Example:
        prompt: iobenzamic acid
        updates: [[iobenzamic acid, is a, drug], [iobenzamic acid, may have, side effects],
        [side effects, can include, nausea], [iobenzamic acid, used as, X-ray contrast
        agent], [iobenzamic acid, formula, C16H13I3N2O3]]
        """

    prompt = f"""
    Given a prompt (a medical condition/procedure/drug), extrapolate as many
    relationships as possible of it and provide a list of updates.
    The relationships should be helpful for healthcare prediction (e.g., drug
    recommendation, mortality prediction, readmission prediction …)
    Each update should be exactly in format of [ENTITY 1, RELATIONSHIP, ENTITY 2]. The
    relationship is directed, so the order matters.
    Both ENTITY 1 and ENTITY 2 should be noun.
    Any element in [ENTITY 1, RELATIONSHIP, ENTITY 2] should be conclusive, make it as
    short as possible.
    Do this in both breadth and depth. Expand [ENTITY 1, RELATIONSHIP, ENTITY 2] until
    the size reaches 100.
    {example}
    prompt: {term}
    updates:
    """
    return prompt

if __name__ == '__main__':
    # print(ollama.invoke("Write an essey about why the sky is blue"))
    kg = get_kg_from_llm("Autism", "condition")
    print(kg)