import json
import stanza
import pandas as pd

DOWNLOAD_RESOURCES = False


def split_documents(dataset_files: dict, tokenizers: dict):
    for dataset in dataset_files:
        for lang in dataset_files[dataset]:
            for file in dataset_files[dataset][lang]:
                sentences = split_document(file['raw'], tokenizers[lang])
                annotated_document = annotate_document(sentences, file['annotated'])
                break
            break
        break


def split_document(document_path: str, tokenizer):
    document_lines = open(document_path, encoding='utf-8-sig').readlines()
    content = ' '.join(document_lines[5:])
    doc = tokenizer(content)
    sentences = doc.sentences
    return sentences


def annotate_document(sentences, annotations_path):
    print(annotations_path)
    annotations = pd.read_csv(annotations_path, names=['Mention', 'Base', 'Category', 'clID'], skiprows=[0], sep='\t').to_dict('records')
    for annotation in annotations:
        match_ids = []
        ann_pieces = annotation['Mention'].split()
        for sentence in sentences:
            for token in sentence.tokens:
                found_id = ann_pieces.index(token.text.lower())
                if found_id >= 0:
                    print(f'MATCH: {token.text}, id: {found_id}')
                    match_ids.append(token.id)
                if len(ann_pieces) == len(match_ids):
                    break

            print(sentence)
            break
        break
    return []


if __name__ == '__main__':
    datasets_files = json.load(open('./data/results/dataset_pairs.json'))
    languages = set([lang for dataset in datasets_files for lang in datasets_files[dataset].keys()])
    print(languages)
    if DOWNLOAD_RESOURCES: # do it once on a new system
        for lang in languages:
            lang = lang if lang != 'ua' else 'uk'
            print(f'Downloading {lang}...')
            stanza.download(lang, processors='tokenize')
    tokenizers = {lang: stanza.Pipeline(lang=lang if lang != 'ua' else 'uk', processors='tokenize') for lang in languages}
    split_documents(datasets_files, tokenizers)
