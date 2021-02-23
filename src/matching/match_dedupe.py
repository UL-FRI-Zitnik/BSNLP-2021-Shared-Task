import json
import pathlib

from dedupe import Dedupe, StaticDedupe, console_label
from datetime import datetime
from collections import defaultdict
from itertools import combinations, product
from random import choices, random
from typing import Iterable, Callable

from src.matching.match import load_nes

BASE_FNAME = "./data/deduper"

# Dedup configuration variables
CHOOSE_K = 3  # determines how many samples of equivalent values to choose
CLUSTER_THRESHOLD = 0.35
DEDUPE_CORES_USED = 14
dedupe_variables = [
    # document structure: docId,sentenceId,tokenId,text,lemma,calcLemma,upos,xpos,ner,clID
    # variables to consider:
    {"field": "text", "type": "String"},
    {"field": "calcLemma", "type": "String"},
    {"field": "upos", "type": "String"},
    {"field": "xpos", "type": "String"},
    {"field": "ner", "type": "String"},  # this has to use the predicted NE tags
]


def load_data(
    clear_cache: bool = False
) -> dict:
    cache_path = f'{BASE_FNAME}/cached_data.json'
    cached_file = pathlib.Path(cache_path)
    if not clear_cache and cached_file.exists() and cached_file.is_file():
        mod_time = datetime.fromtimestamp(cached_file.stat().st_mtime)
        print(f"Using cached data from `{cache_path}`, last modified at: `{mod_time.isoformat()}`")
        with open(cache_path) as f:
            return json.load(f)
    # load the datasets
    datasets = json.load(open("./data/results/dataset_pairs.json"))
    data = load_nes(datasets, filter_nes=True, flatten_docs=True)

    # transform the data from a list to a dictionary
    ret = {}
    for dataset, langs in data.items():
        dataset_name = dataset.split('/')[-1]
        ret[dataset_name] = {}
        for lang, items in langs.items():
            ret[dataset_name][lang] = {}
            for item in items:
                ret[dataset_name][lang][f"{item['docId']}-{item['sentenceId']}-{item['text']}"] = item

    # store a cached version of the data
    with open(cache_path, 'w') as f:
        print(f"Storing cached data at: {cache_path}")
        json.dump(ret, f)

    return ret


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
    data: dict
) -> dict:
    positive_examples = defaultdict(list)
    matches = []
    distinct = []

    for key, value in data.items():
        positive_examples[value['clID']].append(value)

    for key, values in positive_examples.items():
        # print(f"{key} ({len(values)}): {values}")
        use_items = choices(values, k=3)
        for comb in combinations(use_items, 2):
            matches.append(comb)

    # TODO: find most similar mentions and generate negative
    #   similarity search

    clids = positive_examples.keys()
    for comb in combinations(clids, 2):
        # skip some combination with a 1/2 probability
        if random() < 0.5:
            continue
        d1 = choices(positive_examples[comb[0]], k=CHOOSE_K)
        d2 = choices(positive_examples[comb[1]], k=CHOOSE_K)
        for items in product(d1, d2):
            distinct.append(items)

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
                if lang != 'sl':
                    continue
                call_fun(dataset, lang, items)
    return loop_through


def train(
    dataset: str,
    lang: str,
    items: dict
) -> None:
    print(f"Training on `{dataset}`, language: `{lang}`")

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
    print(f"Clustering `{dataset}`, language: `{lang}`")

    learned_settings_fname = f'{RUN_BASE_FNAME}/learned_settings-{dataset}-{lang}.bin'
    settings_file = pathlib.Path(learned_settings_fname)
    if not (settings_file.exists() or settings_file.is_file()):
        print(f"Settings file `{learned_settings_fname}` does not exist or it's not a file.")
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
            print(row)
            print(row, file=f)

    clustered_data_fname = f'{RUN_BASE_FNAME}/clusters-{dataset}-{lang}.json'
    clusters = get_clustered_ids(clustered)
    with open(clustered_data_fname, 'w') as f:
        json.dump(clusters, fp=f, indent=4)


def main():
    print("Running Dedupe Entity Matching")

    print("Loading the data...")
    data = load_data()

    print("Training on the data...")
    trainer = data_looper(data, train)
    trainer()

    print("Clustering the data...")
    clusterer = data_looper(data, cluster_data)
    clusterer()

    print("Done!")


if __name__ == '__main__':
    run_time = datetime.now().isoformat()[:-7]  # exclude the ms
    RUN_BASE_FNAME = f"{BASE_FNAME}/runs/run_{run_time}"
    pathlib.Path(RUN_BASE_FNAME).mkdir(parents=True, exist_ok=True)
    main()
