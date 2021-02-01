import json
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from typing import Callable, Union
from collections import defaultdict, Counter
from itertools import combinations
from src.utils.utils import list_dir

node_att_list = ["docId", "sentenceId", "tokenId", "text", "lemma", "ner", "clID", "lang", "numTokens", "contracted"]
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
                df['contracted'] = 0
                filtered = df.loc[~df['ner'].isin(['O'])].to_dict(orient='records')
                document = merge_ne_records(filtered)
                documents.append(document)
            break
        break
    return documents


def get_atts(
    att_fun: Callable,
    graph: nx.Graph,
    att_list: list
) -> Callable:
    """
        Gets the attributes for a given label.
        2-steps:
            1. define the attributes
            2. get the attributes for the label
    :param att_fun
    :param graph
    :param att_list: attribute list
    :return:
    """
    atts = {}
    for att in att_list:
        atts[att] = att_fun(graph, att)

    def get(label: str) -> dict:
        return {att: atts[att][label] for att in att_list}
    return get


def print_graph(
    g: nx.Graph,
    print_details: bool = False,
    draw_labels: bool = False
) -> None:
    get_node_atts = get_atts(nx.get_node_attributes, g, node_att_list)
    get_edge_atts = get_atts(nx.get_edge_attributes, g, edge_att_list)

    print(f"Nodes: {len(g.nodes)}")
    print(f"Edges: {len(g.edges)}")
    print(f"FCCs: {len(list(nx.connected_components(g)))}")
    if print_details:
        for n in g.nodes:
            print(f"Node: {n}, atts: {get_node_atts(n)}")
        for e in g.edges:
            print(f"Edge: {e}, atts: {get_edge_atts(e)}")
    pos = nx.kamada_kawai_layout(g)
    nx.draw(
        g,
        pos,
        with_labels=draw_labels,
        width=[g[u][v]['weight'] / 10 for u, v in g.edges]
    )
    plt.show()


def contract_document_nes(g: nx.Graph) -> (nx.Graph, dict):
    """
        Contract document named entities based on their attributes:
            - [x] lemmas
            - [ ]
    :param g:
    :return: contracted graph, dictionary of contracted nodes
    """
    def get_lemma_counts(graph: nx.Graph) -> dict:
        get_lemmas = get_atts(nx.get_node_attributes, graph, ['lemma'])
        lemmas = defaultdict(list)
        for n in graph.nodes:
            lemmas[get_lemmas(n)['lemma']].append(n)
        return dict(lemmas)
    get_node_atts = get_atts(nx.get_node_attributes, g, node_att_list)

    # a copy of the graph to contract
    contracted_graph = g.copy()
    contracted_nodes = defaultdict(list)
    lemmas = get_lemma_counts(contracted_graph)
    print(json.dumps(lemmas, indent=4))
    for lemma, nids in lemmas.items():
        contract_into = nids[0]
        for to_contract in nids[1:]:
            contracted_graph = nx.contracted_nodes(contracted_graph, contract_into, to_contract, copy=False)
            contracted_graph.nodes[contract_into]['contracted'] += 1
            contracted_nodes[contract_into].append(get_node_atts(to_contract))
    return contracted_graph, contracted_nodes


def build_graph(documents: list) -> (nx.Graph, dict):
    def get_node_id(ne: dict):
        return f"{ne['docId']}-{ne['sentenceId']}-{ne['lemma'].replace(' ', '_')}"

    def add_edges(nodes: list, g: nx.Graph):
        for u, v in combinations(nodes, 2):
            if (u, v) in g.edges or (v, u) in g.edges:
                g.edges[(u, v)]['weight'] += 1
            else:
                g.add_edge(u, v, weight=1)

    # dataset graph
    ds_graph = nx.Graph()
    contracted_nodes = {}
    i = 0
    for document in documents:
        i += 1
        doc_graph = nx.Graph()
        document_nodes = []
        sentence_ids = defaultdict(list)
        # add all the NEs to a document graph
        for ne in document:
            nid = get_node_id(ne)
            document_nodes.append(nid)
            doc_graph.add_node(nid, **ne)
            sentence_ids[ne['sentenceId']].append(nid)

        # connect the NEs from the same sentences
        for sentence, items in sentence_ids.items():
            add_edges(items, doc_graph)

        print_graph(doc_graph, print_details=False)

        # contract the NEs within the same document
        doc_graph, contracted = contract_document_nes(doc_graph)

        # merge all the contracted nodes
        contracted_nodes = {**contracted_nodes, **contracted}

        # add edges between all NEs within the document to form a clique
        # TODO: reconsider this
        add_edges(list(doc_graph.nodes), doc_graph)

        # merge the document graph with the dataset graph
        ds_graph.add_nodes_from(doc_graph.nodes(data=True))
        ds_graph.add_edges_from(doc_graph.edges(data=True))

        if i == 2:  # early stopping, for debugging purposes TODO: delete me
            break
    # TODO: merge entities across documents
    return ds_graph, contracted_nodes


def main():
    datasets = json.load(open("./data/results/dataset_pairs.json"))
    doc_nes = load_nes(datasets)
    # print(json.dumps(doc_nes, indent=4))
    g, contracted = build_graph(doc_nes)
    print_graph(g, print_details=False, draw_labels=True)
    print(json.dumps(contracted, indent=4))


if __name__ == '__main__':
    main()
