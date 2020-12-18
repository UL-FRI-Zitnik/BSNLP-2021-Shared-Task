import json
import os
import stanza
import pandas as pd

DOWNLOAD_RESOURCES = False


def split_documents(dataset_files: dict, tokenizers: dict):
    warnings = []
    files_processed = 0
    for dataset in dataset_files:
        for lang in dataset_files[dataset]:
            print(f'Dataset: {dataset}, Language: {lang}')
            merged_path = f'{dataset}/merged/{lang}'
            if not os.path.exists(merged_path):
                os.mkdir(merged_path)
            for file in dataset_files[dataset][lang]:
                sentences, document_id = split_document(file['raw'], tokenizers[lang])
                annotated_document, warns = annotate_document(sentences, file['annotated'], document_id)
                warnings.extend(warns)
                doc_name = f"{file['raw'].split('/')[-1][:-3]}csv"
                merged_fname = f'{merged_path}/{doc_name}'
                annotated_document.to_csv(merged_fname, index=False)
                files_processed += 1
    print(f'Files processed: {files_processed}.')
    print(f'Number of warnings occured: {len(warnings)}.')
    json.dump(warnings, open('./data/results/merge_warnings.json', 'w'), indent=4)


def split_document(document_path: str, tokenizer):
    document_lines = open(document_path, encoding='utf-8-sig').readlines()
    document_id = document_lines[0].strip()
    content = ' '.join(document_lines[4:])
    doc = tokenizer(content)
    sentences = [sentence.to_dict() for sentence in doc.sentences]
    return sentences, document_id


def annotate_document(sentences: list, annotations_path: str, document_id: str) -> (pd.DataFrame, list):
    annotations = pd.read_csv(annotations_path, names=['Mention', 'Base', 'Category', 'clID'], skiprows=[0], sep='\t').to_dict('records')
    annotation = annotations.pop(0)
    match_ids = []
    ann_pieces = annotation['Mention'].split()
    annotated_tokens = []
    warnings = []
    for sent_id, sentence in enumerate(sentences):
        for token in sentence:
            try:
                found_id = ann_pieces.index(token['text'].lower())
                match_ids.append(token['id'])
            except ValueError:
                token['ner'] = 'O'
                token['lemma'] = ''
                token['clID'] = ''
            if len(ann_pieces) == len(match_ids):
                f_ner = True
                lemma = annotation["Base"].split()
                for t in sentence:
                    if t['id'] in match_ids:
                        t['ner'] = f"{'B' if f_ner else 'I'}-{annotation['Category']}"
                        if not lemma:
                            warnings.append({"msg": "BASE FORM DOES NOT MATCH MENTION", "doc": annotations_path, "lemma": annotation['Base'], "ner": annotation['Mention']})
                            print(f"[WARNING] LEMMA DOES NOT MATCH")
                            lemma = ['PAD']
                        t['lemma'] = lemma.pop(0)
                        t['clID'] = annotation["clID"]
                        f_ner = False if f_ner else f_ner
                if annotations:
                    annotation = annotations.pop(0)
                    match_ids = []
                    ann_pieces = annotation['Mention'].split()
                else:
                    match_ids = []
                    ann_pieces = [0]  # hack to prevent entering the if
            token['sentenceId'] = sent_id
            token['docId'] = document_id
        annotated_tokens.extend(sentence)

    if annotations:
        anns = [annotation]
        anns.extend(annotations)
        warnings.append({
            "msg": "Annotations are not empty",
            "doc": annotations_path,
            "tokens": annotated_tokens,
            "annotations": annotations
        })
        print(f"[WARNING] ANNOTATIONS LEFT")
    sentence_df = pd.DataFrame(annotated_tokens)
    sentence_df = sentence_df.rename(columns={'id': 'tokenId'})
    sentence_df = sentence_df[['docId', 'sentenceId', 'tokenId', 'text', 'lemma', 'ner', 'clID']]  # leaving out 'misc' for now
    return sentence_df, warnings


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
