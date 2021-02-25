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
            print(f'Dataset: {dataset}, Language: {lang}')
            merged_path = f'{dataset}/merged/{lang}'
            if not os.path.exists(merged_path):
                os.mkdir(merged_path)
            for file in dataset_files[dataset][lang]:
                sentences, document_id = split_document(file['raw'], tokenizers[lang], lang)
                annotated_document, warns = annotate_document(sentences, file['annotated'], document_id, tokenizers[lang], lang)
                warnings.extend(warns)
                doc_name = f"{file['raw'].split('/')[-1][:-3]}csv"
                merged_fname = f'{merged_path}/{doc_name}'
                annotated_document.to_csv(merged_fname, index=False)
                files_processed += 1
    print(f'Files processed: {files_processed}.')
    print(f'Number of warnings occured: {len(warnings)}.')
    json.dump(warnings, open('./data/results/merge_warnings.json', 'w'), indent=4)


def convert_sentences(raw_sentences, lang):
    sentences = []
    for sentence in raw_sentences:
        tokens = []
        for token in sentence.tokens:
            if len(token.words) > 1:
                print(f"MORE WORDS: {token.words}")
            tokens.append({
                "id": token.index if lang in ['sl', 'bg'] else token.id[0],
                "text": ''.join([w.text for w in token.words]),
                "calcLemma": ' '.join([w.lemma for w in token.words if w.lemma is not None]),
                "upos": ' '.join([w.xpos for w in token.words if w.xpos is not None]),
                "xpos": ' '.join([w.upos for w in token.words if w.upos is not None]),
            })
        sentences.append(tokens)
    return sentences


def split_document(document_path: str, tokenizer, lang: str):
    document_lines = open(document_path, encoding='utf-8-sig').readlines()
    document_id = document_lines[0].strip()
    content = ' '.join(document_lines[4:])
    doc = tokenizer(content)
    # sentences = [sentence.to_dict() for sentence in doc.sentences] if lang != 'sl' else convert_sentences(doc.sentences)
    sentences = convert_sentences(doc.sentences, lang)
    return sentences, document_id


def tokenize_mention(mention: str, tokenizer, lang: str) -> list:
    # just for slo
    tokenized = [i for s in convert_sentences(tokenizer(mention).sentences, lang) for i in s]
    return [t['text'] for t in tokenized]


def sort_by_mention_length(data: pd.DataFrame) -> pd.DataFrame:
    sorted_vals = data['Mention'].str.len().sort_values().index
    return data.reindex(sorted_vals).reset_index(drop=True)


def annotate_document(sentences: list, annotations_path: str, document_id: str, tokenizer, lang) -> (pd.DataFrame, list):
    # print(tf"Work on {annotations_path}")
    try:
        anns = pd.read_csv(annotations_path, names=['Mention', 'Base', 'Category', 'clID'], skiprows=[0], sep='\t')
    except:
        print(f"CAN'T LOAD {annotations_path}")
        return pd.DataFrame(), []
    # a hack to first look for shorter matches if mentions
    # are substrings, e.g. komisija vs Evropska Komisija
    ann_df = sort_by_mention_length(anns)

    warnings = []
    if len(ann_df['Mention'].unique()) != len(ann_df.index):
        print("Duplicate mentions!")
        warnings.append({
            "msg": "Duplicate mentions found!",
            "doc": annotations_path,
        })
    annotations = ann_df.to_dict('records')
    annotated_tokens = []
    for sent_id, sentence in enumerate(sentences):
        for token in sentence:
            token['ner'] = 'O'
            token['lemma'] = ''
            token['clID'] = ''
            token['sentenceId'] = sent_id
            token['docId'] = document_id
            annotated_tokens.append(token)

    used_annotations = 0
    for annotation in annotations:
        ann_pieces = tokenize_mention(annotation['Mention'], tokenizer, lang)
        matched = 0
        for token_id, token in enumerate(annotated_tokens):
            first_ratio = fuzz.ratio(ann_pieces[0].lower(), token['text'].lower())
            if first_ratio >= LOWEST_SIMILARITY:
                if token_id + len(ann_pieces) > len(annotated_tokens):
                    continue
                all_ratio = [fuzz.ratio(ann.lower(), annotated_tokens[token_id + i]['text'].lower()) for i, ann in enumerate(ann_pieces)]
                if len([r for r in all_ratio if r >= LOWEST_SIMILARITY]) != len(ann_pieces):
                    continue
                f_ner = True
                matched_tokens = [annotated_tokens[token_id + i]['text'] for i, _ in enumerate(ann_pieces)]
                lemma = tokenize_mention(str(annotation["Base"]), tokenizer, lang)
                for i, _ in enumerate(ann_pieces):
                    t = annotated_tokens[token_id + i]
                    t['ner'] = f"{'B' if f_ner else 'I'}-{annotation['Category']}"
                    if not lemma:
                        warnings.append({
                            "msg": "BASE FORM DOES NOT MATCH MENTION",
                            "doc": annotations_path,
                            "lemma": annotation['Base'],
                            "ner": annotation['Mention'],
                            "matched": matched_tokens
                        })
                        print(f"[WARNING] LEMMA DOES NOT MATCH")
                        lemma = ['PAD']
                    t['lemma'] = lemma.pop(0)
                    t['clID'] = annotation["clID"]
                    f_ner = False if f_ner else f_ner
                matched += 1
        if matched == 0:
            warnings.append({
                "msg": "Annotation not matched!",
                "doc": annotations_path,
                "annotation": annotation,
            })
        used_annotations += 1 if matched > 0 else 0

    if used_annotations != len(annotations):
        print(f"[WARNING] UNUSED ANNOTATIONS: {used_annotations}/{len(annotations)}")
        warnings.append({
            "msg": f"ALTERED ITEMS ({used_annotations}) NOT EQUAL TO ANNOTATIONS ({len(annotations)})",
            "doc": annotations_path,
            "num_altered": used_annotations,
            "num_annotations": len(annotations)
        })
    sentence_df = pd.DataFrame(annotated_tokens)
    sentence_df = sentence_df.rename(columns={'id': 'tokenId'})
    sentence_df = sentence_df[['docId', 'sentenceId', 'tokenId', 'text', 'lemma', 'calcLemma', 'upos', 'xpos', 'ner', 'clID']]  # leaving out 'misc' for now
    return sentence_df, warnings


if __name__ == '__main__':
    datasets_files = json.load(open('./data/results/dataset_pairs.json'))
    languages = set([lang for dataset in datasets_files for lang in datasets_files[dataset].keys()])
    print(languages)
    processors = 'tokenize,pos,lemma'
    if DOWNLOAD_RESOURCES:  # do it once on a new system
        for lang in languages:
            lang = lang if lang != 'ua' else 'uk'
            print(f'Downloading {lang}...')
            stanza.download(lang, processors=processors)
        classla.download('sl')
        classla.download('bg')
    tokenizers = {lang: stanza.Pipeline(lang=lang if lang != 'ua' else 'uk', processors=processors) for lang in languages}
    tokenizers['sl'] = classla.Pipeline('sl', processors=processors)
    tokenizers['bg'] = classla.Pipeline('bg', processors=processors)
    split_documents(datasets_files, tokenizers)
