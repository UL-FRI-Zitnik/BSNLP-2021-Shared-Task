import sys
import json
import glob
import pathlib
import logging
import argparse
import pandas as pd

from collections import defaultdict

from src.utils.load_dataset import LoadBSNLP
from src.utils.load_documents import LoadBSNLPDocuments
from src.utils.update_documents import UpdateBSNLPDocuments
from src.utils.utils import list_dir

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
)
logger = logging.getLogger('TrainEvalModels')

# pred_path = 'data/runs/run_2497_multilang_all'
pred_path = './data/runs/run_l1o_2551'
cluster_path = 'data/deduper/runs/run_2508'


def load_clusters(
    path: str
) -> (dict, dict):
    clusters = {}
    ne_map = {}
    n_clusters = 0
    for dataset in LoadBSNLP.datasets['test_2021']:
        df_clusters = pd.DataFrame()
        ne_map[dataset] = defaultdict(list)
        for fname in glob.glob(f'{path}/{dataset}/clusters-*.json'):
            fcluster = json.load(open(fname))
            nes = []
            for cluster in fcluster:
                for ne in cluster['ners']:
                    try:
                        ids = ne['id'].split(';')
                        for sid, tid, t in zip(ids[2].split(':'), ids[3].split(':'), ids[4].split(' ')):
                            item = {
                                'clusterId': f'{n_clusters}-{cluster["clusterId"]}',
                                'lang': ids[0],
                                'docId': ids[1],
                                'sentenceId': int(sid),
                                'tokenId': int(tid),
                                'text': t,
                            }
                            ne_key = f'{ids[0]};{ids[1]};{sid};{tid}'
                            if ne_key in ne_map[dataset]:
                                logger.info(f"Double occurrence: {ne_key}")
                            ne_map[dataset][ne_key].append(f'{n_clusters}-{cluster["clusterId"]}')
                            nes.append(item)
                    except Exception as e:
                        logger.error(f"ERROR OCCURRED {ne}, {e}")
            n_clusters += 1
            df_clusters = pd.concat([df_clusters, pd.DataFrame(nes)])
        clusters[dataset] = df_clusters
    logger.info(f"Clusters: {clusters}")
    logger.info(f"Map: {ne_map}")
    return clusters, ne_map


def update_clusters(data: dict, ne_map: dict):
    for dataset, langs in data.items():
        missed = 0
        all_nes = 0
        for lang, docs in langs.items():
            for docId, doc in docs.items():
                doc['content']['calcClId'] = 'xxx'
                for i, row in doc['content'].iterrows():
                    if row['calcNER'] != 'O':
                        all_nes += 1
                    ne_key = f'{lang};{row["docId"]};{row["sentenceId"]};{row["tokenId"]}'
                    if ne_key not in ne_map[dataset]:
                        if row['calcNER'] != 'O':
                            missed += 1
                        continue
                    doc['content'].loc[i, 'calcClId'] = ne_map[dataset][ne_key][0]
        logger.info(f"[{dataset}] Missed {missed}/{all_nes} [{missed/all_nes:.3f}]")
    return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-path', type=str, default=None)
    parser.add_argument('--cluster-path', type=str, default=None)
    parser.add_argument('--year', type=str, default='2021')
    parser.add_argument('--lang', type=str, default='all')
    return parser.parse_args()


def main():
    global pred_path, cluster_path
    args = parse_args()
    pred_path = args.pred_path if args.pred_path is not None else pred_path
    cluster_path = args.cluster_path if args.cluster_path is not None else cluster_path
    year = args.year
    lang = args.lang

    logger.info(f"Predictions path: {pred_path}")
    logger.info(f"Clusters path: {pred_path}")
    logger.info(f"Year: {year}")
    logger.info(f"Language: {lang}")

    path = pathlib.Path(pred_path)
    if not path.exists() or not path.is_dir():
        raise Exception(f"Path does not exist or is not a directory: `{pred_path}`")
    path = pathlib.Path(cluster_path)
    if not path.exists() or not path.is_dir():
        raise Exception(f"Path does not exist or is not a directory: `{cluster_path}`")

    logger.info("Loading the clusters...")
    clusters, ne_map = load_clusters(cluster_path)

    models, _ = list_dir(f'{pred_path}/predictions/bsnlp')
    for model in models:
        logger.info(f"Loading the documents for model `{model}`...")
        data = LoadBSNLPDocuments(year='test_2021', lang=lang, path=f'{pred_path}/predictions/bsnlp/{model}').load_predicted()

        logger.info(f"[{model}] Merging the cluster data into the prediction data")
        updated = update_clusters(data, ne_map)

        logger.info(f"[{model}] Persisting the changes...")
        UpdateBSNLPDocuments(year='test_2021', lang=lang, path=f'{pred_path}/predictions/bsnlp/{model}').update_clustered(updated)

    logger.info("Done.")


if __name__ == '__main__':
    main()
