import json
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from typing import Callable
from collections import defaultdict
from itertools import combinations
from src.utils.utils import list_dir

node_att_list = ["docId", "sentenceId", "tokenId", "text", "lemma", "ner", "clID", "lang", "numTokens"]
edge_att_list = ["weight"]


def merge_ne_records(nes: list) -> list:
    merged = []
    for i, ne in enumerate(nes):
        if ne['ner'].startswith('I-'):
            continue
        j = i + 1
        while j < len(nes) and not nes[j]['ner'].startswith('B-'):
            if nes[j]['tokenId'] != (nes[j - 1]['tokenId'] + 1):
                raise Exception("Tokens are not coming one after the other")
            ne['text'] += f' {nes[j]["text"]}'
            ne['lemma'] += f' {nes[j]["lemma"]}'
            ne['numTokens'] += 1
            j += 1
        ne['ner'] = ne['ner'][2:]
        merged.append(ne)
    return merged


def load_nes(datasets):
    documents = []
    for dataset, langs in datasets.items():
        print(f'Extracting from: {dataset}')
        for lang in langs.keys():
            # focus on slovenian for now
            if lang != 'sl':
                continue
            ne_path = f'{dataset}/merged/{lang}'
            _, files = list_dir(ne_path)
            for file in files:
                df = pd.read_csv(f'{ne_path}/{file}', dtype={'docId': str, 'clID': str})
                df['lang'] = lang
                df['numTokens'] = 1
                filtered = df.loc[~df['ner'].isin(['O'])].to_dict(orient='records')
                document = merge_ne_records(filtered)
                documents.append(document)
            break
        break
    return documents


def get_atts(atts: dict) -> Callable:
    """
        Gets the attributes for a given label.
        2-steps:
            1. define the attributes
            2. get the attributes for the label
    :param atts:
    :return:
    """
    def get_atts(label: str) -> dict:
        return {att: atts[att][label] for att in atts.keys()}
    return get_atts


def print_graph(
    G: nx.Graph,
    print_details: bool = False,
    draw_labels: bool = False
) -> None:
    node_atts = {}
    edge_atts = {}
    for att in node_att_list:
        node_atts[att] = nx.get_node_attributes(G, att)
    for att in edge_att_list:
        edge_atts[att] = nx.get_edge_attributes(G, att)

    get_node_atts = get_atts(node_atts)
    get_edge_atts = get_atts(edge_atts)

    print(f"Nodes: {len(G.nodes)}")
    print(f"Edges: {len(G.edges)}")
    if print_details:
        for n in G.nodes:
            print(f"Node: {n}, atts: {get_node_atts(n)}")
        for e in G.edges:
            print(f"Edge: {e}, atts: {get_edge_atts(e)}")
    pos = nx.kamada_kawai_layout(G)
    nx.draw(
        G,
        pos,
        with_labels=draw_labels,
        width=[G[u][v]['weight']/10 for u, v in G.edges]
    )
    plt.show()


def build_graph(documents: list):
    def get_node_id(ne: dict):
        return f"{ne['docId']}-{ne['sentenceId']}-{ne['lemma'].replace(' ', '_')}"

    def add_edges(nodes: list, g: nx.Graph):
        for u, v in combinations(nodes, 2):
            if (u, v) in g.edges or (v, u) in g.edges:
                g.edges[(u, v)]['weight'] += 1
            else:
                g.add_edge(u, v, weight=1)

    G = nx.Graph()
    i = 0
    for document in documents:
        i += 1
        document_nodes = []
        sentence_ids = defaultdict(list)
        for ne in document:
            nid = get_node_id(ne)
            document_nodes.append(nid)
            G.add_node(nid, **ne)
            sentence_ids[ne['sentenceId']].append(nid)
        for sentence, items in sentence_ids.items():
            add_edges(items, G)
        add_edges(document_nodes, G)
        # TODO: merge same entities within document
        if i == 2:
            break
    # TODO: merge entities between documents
    print_graph(G)
    return


def main():
    datasets = json.load(open("./data/results/dataset_pairs.json"))
    doc_nes = load_nes(datasets)
    # print(json.dumps(doc_nes, indent=4))
    build_graph(doc_nes)


if __name__ == '__main__':
    main()
