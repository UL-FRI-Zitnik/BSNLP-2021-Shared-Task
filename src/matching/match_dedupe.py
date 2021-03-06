import sys
import os
import argparse
import json
import pathlib
import pandas as pd
import logging

from tqdm import tqdm
from dedupe import Dedupe, StaticDedupe, console_label
from fuzzywuzzy import fuzz
from datetime import datetime
from collections import defaultdict
from itertools import combinations, product
from random import choices, random
from typing import Iterable, Callable

from src.utils.utils import list_dir


logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
)
logger = logging.getLogger('DedupeMatching')

BASE_FNAME: str = "./data/deduper"
run_time = datetime.now().isoformat()[:-7]  # exclude the ms
JOB_ID = os.environ['SLURM_JOB_ID'] if 'SLURM_JOB_ID' in os.environ else run_time
RUN_BASE_FNAME = f"{BASE_FNAME}/runs/run_{JOB_ID}"
DATA_PATH = f"./data/datasets/bsnlp"
NER_FIELD = 'calcNER'
RELEVANT_LANGS: list = ['bg', 'cs', 'pl', 'ru', 'sl', 'uk']

# Dedup configuration variables
SEARCH_CLOSEST: bool = True
CHOOSE_K: int = 2  # determines how many samples of equivalent values to choose
CLUSTER_THRESHOLD: float = 0.65
DEDUPE_CORES_USED: int = 63
dedupe_variables: list = [
    # document structure: docId,sentenceId,tokenId,text,lemma,calcLemma,upos,xpos,ner,clID
    # variables to consider:
    {"field": "text", "type": "String"},
    {"field": "calcLemma", "type": "String"},
    {"field": "upos", "type": "String"},
    {"field": "xpos", "type": "String"},
    {"field": "ner", "type": "String"},
]


def merge_nes(
    nes: list
) -> list:
    """
        Merges the NEs in the form of the expected output
    :param nes:
    :return:
    """
    merged = []
    for i, ne in enumerate(nes):
        if ne[NER_FIELD].startswith('I-'):
            continue
        j = i + 1
        ne['numTokens'] = 1
        while j < len(nes) and not nes[j][NER_FIELD].startswith('B-'):
            ne['text'] = f'{ne["text"]} {nes[j]["text"]}'
            ne['lemma'] = f'{ne["lemma"]} {nes[j]["lemma"]}'
            ne['calcLemma'] = f'{ne["calcLemma"]} {nes[j]["calcLemma"]}'
            ne['sentenceId'] = f'{ne["sentenceId"]}:{nes[j]["sentenceId"]}'
            ne['tokenId'] = f'{ne["tokenId"]}:{nes[j]["tokenId"]}'
            ne['upos'] = f'{ne["upos"]}:{nes[j]["upos"]}'
            ne['xpos'] = f'{ne["xpos"]}:{nes[j]["xpos"]}'
            if nes[j]["clID"] != ne['clID']:
                print(f"Inconsistent cluster ids: {nes[j]['clID']} vs {ne['clID']}, NE: {ne}")
            ne['numTokens'] += 1
            j += 1
        ne[NER_FIELD] = ne[NER_FIELD][2:]
        merged.append(ne)
    return merged


def load_nes(
    datasets: list,
) -> (dict, dict):
    documents = {}
    doc_alphabet = {}
    # doc_alphabet = defaultdict(dict)
    for dataset in datasets:
        dataset_name = dataset.split('/')[-1]
        if dataset_name not in ['covid-19', 'us_election_2020']:
            print(f"Skipping {dataset_name}")
            continue
        documents[dataset_name] = {}
        doc_alphabet[dataset_name] = defaultdict(dict)
        langs, _ = list_dir(f'{dataset}/predicted')
        for lang in langs:
            if lang.lower() not in RELEVANT_LANGS:
                logger.info(f"Skipping {dataset_name}/{lang}")
                continue
            documents[dataset_name][lang] = {}
            logger.info(f'Extracting from: {dataset}/{lang}')
            ne_path = f'{dataset}/predicted/{lang}'
            _, files = list_dir(ne_path)
            for file in files:
                df = pd.read_csv(f'{ne_path}/{file}', dtype={'docId': str, 'sentenceId': str, 'tokenId': str, 'clID': str,'text': str,'lemma': str,'calcLemma': str,'upos': str,'xpos': str,'ner': str})
                df['lang'] = lang
                df = df.fillna('N/A')
                records = merge_nes(df.loc[~(df[NER_FIELD] == 'O')].to_dict(orient='records'))
                for item in records:
                    dkey = f"{lang};{item['docId']};{item['sentenceId']};{item['tokenId']};{item['text']}"
                    fchar = item['text'][0].upper()
                    if dkey in doc_alphabet[dataset_name][fchar]:
                        raise Exception(f"[doc_alphabet] COLLISION!!! {dkey}")
                    doc_alphabet[dataset_name][fchar][dkey] = item
                    if dkey in documents[dataset_name][lang]:
                        raise Exception(f"[documents] COLLISION!!! {dkey}")
                    documents[dataset_name][lang][dkey] = item
    return {
        "normal": documents,
        "alphabetized": doc_alphabet,
    }


def load_data(
    clear_cache: bool = False
) -> (dict, dict):
    cache_path = f'{RUN_BASE_FNAME}/cached_data.json'
    cached_file = pathlib.Path(cache_path)
    if not clear_cache and cached_file.exists() and cached_file.is_file():
        mod_time = datetime.fromtimestamp(cached_file.stat().st_mtime)
        logger.info(f"Using cached data from `{cache_path}`, last modified at: `{mod_time.isoformat()}`")
        with open(cache_path) as f:
            return json.load(f)
    # datasets = json.load(open("./data/results/dataset_pairs.json"))
    datasets, _ = list_dir(DATA_PATH)
    datasets = [f'{DATA_PATH}/{dataset}' for dataset in datasets]
    data = load_nes(datasets)
    with open(cache_path, 'w') as f:
        logger.info(f"Storing cached data at: {cache_path}")
        json.dump(data, f)
    return data


def get_clustered_ids(
    clustered: Iterable
) -> list:
    return [{
        "clusterId": i,
        "ners": [
            {
                'id': cid,
                'score': float(score)
            } for cid, score in zip(ids, scores)
        ]
    } for i, (ids, scores) in enumerate(clustered)]


def generate_training_examples(
    data: dict,
) -> dict:
    positive_examples = defaultdict(list)
    matches = []
    distinct = []

    for key, value in data.items():
        positive_examples[value['clID']].append(value)

    for key, values in positive_examples.items():
        # logger.info(f"{key} ({len(values)}): {values}")
        use_items = choices(values, k=CHOOSE_K)
        for comb in combinations(use_items, 2):
            matches.append(comb)

    clids = positive_examples.keys()
    for comb in combinations(clids, 2):
        # skip some combination with a 1/2 probability
        if not SEARCH_CLOSEST and random() < 0.5:  # toss a fair coin
            # logger.info("Skipping...")
            continue
        d1 = choices(positive_examples[comb[0]], k=CHOOSE_K)
        d2 = choices(positive_examples[comb[1]], k=CHOOSE_K)
        for (i1, i2) in product(d1, d2):
            if SEARCH_CLOSEST:
                if fuzz.ratio(i1['text'].lower(), i2['text'].lower()) >= 70:
                # logger.info(f"Similar are: {i1['text']}, {i2['text']}")
                    distinct.append((i1, i2))
            else:
                distinct.append((i1, i2))

    return {
        'distinct': distinct,
        'match': matches
    }


def data_looper(
    data: dict,
    call_fun: Callable,
    mapper: dict,
    train_all: bool = False,
) -> Callable:
    chunk_size = 50
    def loop_through():
        for dataset, langs in data.items():
            for lang, items in langs.items():
                try:
                    logger.info(f"size of items for `{dataset}/{lang}`: {len(items)}")
                    keys = list(items.keys())
                    for i, chunk_keys in enumerate([keys[x:x+chunk_size] for x in range(0, len(keys), chunk_size)]):
                        chunk = {k:items[k] for k in chunk_keys}
                        call_fun(dataset, f'{lang}-{i}', chunk, mapper)
                except Exception as e:
                    logger.error(f"ERROR OCCURED WHEN WORKING ON {dataset}/{lang}, {e}")
            if train_all:
                try:
                    call_fun(dataset, "all", {k:v for lang, docs in langs.items() for k, v in docs.items()})
                except Exception as e:
                    logger.error(f"ERROR OCCURED WHEN WORKING ON {dataset}/all, {e}")
    return loop_through


def train(
    dataset: str,
    lang: str,
    items: dict,
    mapper: dict,
) -> None:
    logger.info(f"Training on `{dataset}/{lang}`")

    # prepare training examples: generate matches and distinct cases
    td = generate_training_examples(items)
    train_path = f'{RUN_BASE_FNAME}/{dataset}'
    pathlib.Path(train_path).mkdir(parents=True, exist_ok=True)
    train_data_fname = f'{train_path}/train-{lang}.json'
    with open(train_data_fname, 'w') as tf:
        json.dump(td, tf)

    ## alternatively, manually label the training data
    ## the above code generates the training examples, so it is automating this step
    # console_label(deduper)

    # create a dedupe instance with chosen variables and number of cores to be used
    deduper = Dedupe(variable_definition=dedupe_variables, num_cores=DEDUPE_CORES_USED)

    # load the training data and prepare for training
    with open(train_data_fname) as tf:
        deduper.prepare_training(data=items, training_file=tf)

    # train the deduper
    deduper.train()

    # store the learned settings
    learned_settings_fname = f'{train_path}/learned_settings-{lang}.bin'
    with open(learned_settings_fname, 'wb') as ts:
        deduper.write_settings(ts)


def cluster_data(
    dataset: str,
    lang: str,
    items: dict,
    mapper: dict
) -> None:
    logger.info(f"Clustering `{dataset}/{lang}`")
    data_set_folder = f'{RUN_BASE_FNAME}/{dataset}/'
    pathlib.Path(data_set_folder).mkdir(parents=True, exist_ok=True)
    lang_id = lang.split('-')[0]
    clusters_report_fname = f'{RUN_BASE_FNAME}/{dataset}/clusters_report-{lang}.txt'
    if pathlib.Path(clusters_report_fname).exists():
        logger.info(f"Dataset: `{dataset}/{lang}` is already processed, skipping...")
        return

    learned_settings_fname = f'{RUN_BASE_FNAME}/{mapper[dataset]}/learned_settings-{lang_id}.bin'
    settings_file = pathlib.Path(learned_settings_fname)
    if not (settings_file.exists() or settings_file.is_file()):
        logger.info(f"Settings file `{learned_settings_fname}` does not exist or it's not a file.")
        return

    # load the learned settings
    with open(learned_settings_fname, 'rb') as f:
        deduper = StaticDedupe(f, num_cores=DEDUPE_CORES_USED)

    # cluster the data
    clustered = deduper.partition(items, threshold=CLUSTER_THRESHOLD)

    with open(clusters_report_fname, 'w') as f:
        for clid, (rec, score) in enumerate(clustered):
            print(f"{clid}: {','.join(rec)}", file=f)

    clustered_data_fname = f'{RUN_BASE_FNAME}/{dataset}/clusters-{lang}.json'
    clusters = get_clustered_ids(clustered)
    with open(clustered_data_fname, 'w') as f:
        json.dump(clusters, fp=f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--closest', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--train-chars', action='store_true')
    parser.add_argument('--train-all', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--run-path', type=str, default=None)
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--tsh', type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    
    global RUN_BASE_FNAME, SEARCH_CLOSEST, CLUSTER_THRESHOLD, JOB_ID, DATA_PATH
    RUN_BASE_FNAME = args.run_path if args.run_path is not None else RUN_BASE_FNAME
    DATA_PATH = args.data_path if args.data_path is not None else DATA_PATH
    pathlib.Path(RUN_BASE_FNAME).mkdir(parents=True, exist_ok=True)
    
    CLUSTER_THRESHOLD = args.tsh if args.tsh is not None else CLUSTER_THRESHOLD

    SEARCH_CLOSEST = args.closest

    logger.info("Running Dedupe Entity Matching")
    logger.info(f"SLURM_JOB_ID = {JOB_ID}")
    logger.info(f"Run path = {RUN_BASE_FNAME}")
    logger.info(f"Number of cores = {DEDUPE_CORES_USED}")
    logger.info(f"Dedupe threshold = {CLUSTER_THRESHOLD}")
    logger.info(f"Choose k = {CHOOSE_K}")
    logger.info(f"Closest string search: {SEARCH_CLOSEST}")
    logger.info(f"Train on chars: {args.train_chars}")
    logger.info(f"Train on all datasets: {args.train_all}")
    logger.info(f"Train: {args.train}")
    logger.info(f"Test: {args.test}")

    logger.info("Loading the data...")
    data = load_data()
    data = data['alphabetized'] if args.train_chars else data['normal']

    predict_from = {
        'covid-19': 'ryanair',
        'us_election_2020': 'brexit',
    }

    trainer = data_looper(data, train, train_all=args.train_all, mapper=predict_from)
    if args.train:
        logger.info("Training on the data...")
        trainer()

    clusterer = data_looper(data, cluster_data, mapper=predict_from)
    if args.test:
        logger.info("Clustering the data...")
        clusterer()

    logger.info("Done!")


if __name__ == '__main__':
    main()
