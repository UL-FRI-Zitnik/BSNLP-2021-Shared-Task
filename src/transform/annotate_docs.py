import json
import os
import stanza
import classla
import pandas as pd
from fuzzywuzzy import fuzz

DOWNLOAD_RESOURCES = False
LOWEST_SIMILARITY = 85


def split_documents(dataset_files: dict, tokenizers: dict):
    warnings = []
    files_processed = 0
    for dataset in dataset_files:
        for lang in dataset_files[dataset]:
            if lang != 'sl':
                continue
            print(f'Dataset: {dataset}, Language: {lang}')
            merged_path = f'{dataset}/merged/{lang}'
            if not os.path.exists(merged_path):
                os.mkdir(merged_path)
            for file in dataset_files[dataset][lang]:
                sentences, document_id = split_document(file['raw'], tokenizers[lang], lang)
                annotated_document, warns = annotate_document(sentences, file['annotated'], document_id, tokenizers[lang])
                warnings.extend(warns)
                doc_name = f"{file['raw'].split('/')[-1][:-3]}csv"
                merged_fname = f'{merged_path}/{doc_name}'
                annotated_document.to_csv(merged_fname, index=False)
                files_processed += 1
    print(f'Files processed: {files_processed}.')
    print(f'Number of warnings occured: {len(warnings)}.')
    json.dump(warnings, open('./data/results/merge_warnings.json', 'w'), indent=4)


def convert_sentences(raw_sentences):
    sentences = []
    for sentence in raw_sentences:
        tokens = []
        for token in sentence.tokens:
            if len(token.words) > 1:
                print(f"MORE WORDS: {token.words}")
            tokens.append({
                "id": token.index,
                "text": ''.join([w.text for w in token.words])
            })
        sentences.append(tokens)
    return sentences


def split_document(document_path: str, tokenizer, lang: str):
    document_lines = open(document_path, encoding='utf-8-sig').readlines()
    document_id = document_lines[0].strip()
    content = ' '.join(document_lines[4:])
    doc = tokenizer(content)
    sentences = [sentence.to_dict() for sentence in doc.sentences] if lang != 'sl' else convert_sentences(doc.sentences)
    return sentences, document_id


def tokenize_mention(mention: str, tokenizer) -> list:
    # just for slo
    tokenized = convert_sentences(tokenizer(mention).sentences)[0]
    return [t['text'].lower() for t in tokenized]


def annotate_document(sentences: list, annotations_path: str, document_id: str, tokenizer) -> (pd.DataFrame, list):
    ann_df = pd.read_csv(annotations_path, names=['Mention', 'Base', 'Category', 'clID'], skiprows=[0], sep='\t')
    if len(ann_df['Mention'].unique()) != len(ann_df.index):
        print("Duplicate mentions!")
    annotations = ann_df.to_dict('records')
    annotated_tokens = []
    warnings = []
    for sent_id, sentence in enumerate(sentences):
        for token in sentence:
            token['ner'] = 'O'
            token['lemma'] = ''
            token['clID'] = ''
            token['sentenceId'] = sent_id
            token['docId'] = document_id
            annotated_tokens.append(token)
    altered_items = 0
    for annotation in annotations:
        ann_pieces = tokenize_mention(annotation['Mention'], tokenizer)
        matched = False
        for token_id, token in enumerate(annotated_tokens):
            first_ratio = fuzz.ratio(ann_pieces[0].lower(), token['text'].lower())
            if first_ratio >= LOWEST_SIMILARITY:
                all_ratio = [fuzz.ratio(ann.lower(), annotated_tokens[token_id + i]['text'].lower()) for i, ann in enumerate(ann_pieces)]
                if len([r for r in all_ratio if r >= LOWEST_SIMILARITY]) != len(ann_pieces):
                    continue
                f_ner = True
                lemma = annotation["Base"].split()
                for i, _ in enumerate(ann_pieces):
                    t = annotated_tokens[token_id + i]
                    t['ner'] = f"{'B' if f_ner else 'I'}-{annotation['Category']}"
                    if not lemma:
                        warnings.append({"msg": "BASE FORM DOES NOT MATCH MENTION", "doc": annotations_path, "lemma": annotation['Base'], "ner": annotation['Mention']})
                        print(f"[WARNING] LEMMA DOES NOT MATCH")
                        lemma = ['PAD']
                    t['lemma'] = lemma.pop(0)
                    t['clID'] = annotation["clID"]
                    f_ner = False if f_ner else f_ner
                altered_items += 1
                matched = True
                break
        if not matched:
            warnings.append({
                "msg": "Annotation not matched!",
                "doc": annotations_path,
                "annotation": annotation,
            })
    if altered_items != len(annotations):
        warnings.append({
            "msg": f"ALTERED ITEMS ({altered_items}) NOT EQUAL TO ANNOTATIONS ({len(annotations)})",
            "doc": annotations_path,
            "num_altered": altered_items,
            "num_annotations": len(annotations)
        })
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
    tokenizers['sl'] = classla.Pipeline('sl', processors='tokenize')
    split_documents(datasets_files, tokenizers)
