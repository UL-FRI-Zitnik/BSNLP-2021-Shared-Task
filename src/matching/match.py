import json
import networkx as nx
import pandas as pd

from src.utils.utils import list_dir


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


def main():
    datasets = json.load(open("./data/results/dataset_pairs.json"))
    doc_nes = load_nes(datasets)
    print(doc_nes)


if __name__ == '__main__':
    main()
