import sys
import argparse
import json
import pathlib
import pandas as pd
import logging

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
RUN_BASE_FNAME = f"{BASE_FNAME}/runs/run_{run_time}"

RELEVANT_LANGS: list = ['bg', 'cs', 'pl', 'ru', 'sl', 'uk']

# Dedup configuration variables
SEARCH_CLOSEST: bool = True
CHOOSE_K: int = 3  # determines how many samples of equivalent values to choose
CLUSTER_THRESHOLD: float = 0.65
DEDUPE_CORES_USED: int = 31
dedupe_variables: list = [
    # document structure: docId,sentenceId,tokenId,text,lemma,calcLemma,upos,xpos,ner,clID
    # variables to consider:
    {"field": "text", "type": "String"},
    {"field": "calcLemma", "type": "String"},
    {"field": "upos", "type": "String"},
    {"field": "xpos", "type": "String"},
    {"field": "ner", "type": "String"},
]


def load_nes(
    datasets: dict,
) -> dict:
    documents = {}
    for dataset, langs in datasets.items():
        dataset_name = dataset.split('/')[-1]
        documents[dataset_name] = {}
        for lang in langs.keys():
            if lang.lower() not in RELEVANT_LANGS:
                logger.info(f"Skipping {dataset_name}/{lang}")
                continue
            documents[dataset_name][lang] = {}
            logger.info(f'Extracting from: {dataset}/{lang}')
            ne_path = f'{dataset}/merged/{lang}'
            _, files = list_dir(ne_path)
            for file in files:
                df = pd.read_csv(f'{ne_path}/{file}', dtype={'docId': str, 'sentenceId': str, 'tokenId': str, 'clID': str,'text': str,'lemma': str,'calcLemma': str,'upos': str,'xpos': str,'ner': str})
                df['lang'] = lang
                df = df.fillna('N/A')
                for item in df.loc[~(df['ner'] == 'O')].to_dict(orient='records'):
                    dkey = f"{lang};{item['docId']};{item['sentenceId']};{item['tokenId']};{item['text']}"
                    if dkey in documents[dataset_name][lang]:
                        raise Exception(f"COLLISION!!! {dkey}")
                    documents[dataset_name][lang][dkey] = item
    return documents


def load_data(
    clear_cache: bool = False
) -> dict:
    cache_path = f'{RUN_BASE_FNAME}/cached_data.json'
    cached_file = pathlib.Path(cache_path)
    if not clear_cache and cached_file.exists() and cached_file.is_file():
        mod_time = datetime.fromtimestamp(cached_file.stat().st_mtime)
        logger.info(f"Using cached data from `{cache_path}`, last modified at: `{mod_time.isoformat()}`")
        with open(cache_path) as f:
            return json.load(f)
    datasets = json.load(open("./data/results/dataset_pairs.json"))
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
        use_items = choices(values, k=3)
        for comb in combinations(use_items, 2):
            matches.append(comb)

    clids = positive_examples.keys()
    for comb in combinations(clids, 2):
        # skip some combination with a 1/2 probability
        if not SEARCH_CLOSEST and random() < 0.5:
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
) -> Callable:
    def loop_through():
        for dataset, langs in data.items():
            for lang, items in langs.items():
                try:
                    call_fun(dataset, lang, items)
                except Exception as e:
                    logger.error(f"ERROR OCCURED WHEN WORKING ON {dataset}/{lang}, {e}")
            try:
                call_fun(dataset, "all", {k:v for lang, docs in langs.items() for k, v in docs.items()})
            except Exception as e:
                logger.error(f"ERROR OCCURED WHEN WORKING ON {dataset}/all, {e}")
    return loop_through


def train(
    dataset: str,
    lang: str,
    items: dict
) -> None:
    logger.info(f"Training on `{dataset}/{lang}`")

    # prepare training examples: generate matches and distinct cases
    td = generate_training_examples(items)
    train_data_fname = f'{RUN_BASE_FNAME}/train-{dataset}-{lang}.json'
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
    learned_settings_fname = f'{RUN_BASE_FNAME}/learned_settings-{dataset}-{lang}.bin'
    with open(learned_settings_fname, 'wb') as ts:
        deduper.write_settings(ts)


def cluster_data(
    dataset: str,
    lang: str,
    items: dict
) -> None:
    logger.info(f"Clustering `{dataset}`")

    learned_settings_fname = f'{RUN_BASE_FNAME}/learned_settings-{dataset}-{lang}.bin'
    settings_file = pathlib.Path(learned_settings_fname)
    if not (settings_file.exists() or settings_file.is_file()):
        logger.info(f"Settings file `{learned_settings_fname}` does not exist or it's not a file.")
        return

    # load the learned settings
    with open(learned_settings_fname, 'rb') as f:
        deduper = StaticDedupe(f, num_cores=DEDUPE_CORES_USED)

    # cluster the data
    clustered = deduper.partition(items, threshold=CLUSTER_THRESHOLD)

    clusters_report_fname = f'{RUN_BASE_FNAME}/clusters_report-{dataset}-{lang}.txt'
    with open(clusters_report_fname, 'w') as f:
        for clid, (rec, score) in enumerate(clustered):
            row = f"{clid}: {','.join(rec)}"
            logger.info(row)
            print(row, file=f)

    clustered_data_fname = f'{RUN_BASE_FNAME}/clusters-{dataset}-{lang}.json'
    clusters = get_clustered_ids(clustered)
    with open(clustered_data_fname, 'w') as f:
        json.dump(clusters, fp=f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--closest', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--run-path', type=str, default=None)
    parser.add_argument('--tsh', type=float, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    
    global RUN_BASE_FNAME, SEARCH_CLOSEST, CLUSTER_THRESHOLD
    RUN_BASE_FNAME = args.run_path if args.run_path is not None else RUN_BASE_FNAME
    pathlib.Path(RUN_BASE_FNAME).mkdir(parents=True, exist_ok=True)
    
    CLUSTER_THRESHOLD = args.tsh if args.tsh is not None else CLUSTER_THRESHOLD

    SEARCH_CLOSEST = args.closest

    logger.info("Running Dedupe Entity Matching")
    logger.info(f"Run path = {RUN_BASE_FNAME}")
    logger.info(f"Number of cores = {DEDUPE_CORES_USED}")
    logger.info(f"Dedupe threshold = {CLUSTER_THRESHOLD}")
    logger.info(f"Choose k = {CHOOSE_K}")
    logger.info(f"Closest string search: {SEARCH_CLOSEST}")

    logger.info("Loading the data...")
    data = load_data()

    trainer = data_looper(data, train)
    if args.train:
        logger.info("Training on the data...")
        trainer()

    clusterer = data_looper(data, cluster_data)
    if args.test:
        logger.info("Clustering the data...")
        clusterer()

    logger.info("Done!")


if __name__ == '__main__':
    main()
