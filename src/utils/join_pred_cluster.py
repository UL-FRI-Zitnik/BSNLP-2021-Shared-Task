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

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
)
logger = logging.getLogger('TrainEvalModels')

pred_path = 'data/runs/run_2497_multilang_all'
cluster_path = 'data/deduper/runs/run_2508'


def load_clusters(
    path: str
) -> (dict, dict):
    clusters = {}
    ne_map = {}
    n_clusters = 0
    for dataset in LoadBSNLP.datasets['2021']:
        df_clusters = pd.DataFrame()
        ne_map[dataset] = defaultdict(list)
        for fname in glob.glob(f'{path}/{dataset}/clusters-*.json'):
            fcluster = json.load(open(fname))
            nes = []
            for cluster in fcluster:
                for ne in cluster['ners']:
                    ids = ne['id'].split(';')
                    for sid, tid, t in zip(ids[2].split(','), ids[3].split(','), ids[4].split(' ')):
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
                            print(f"Double occurrence: {ne_key}")
                        ne_map[dataset][ne_key].append(f'{n_clusters}-{cluster["clusterId"]}')
                        nes.append(item)
            n_clusters += 1
            df_clusters = pd.concat([df_clusters, pd.DataFrame(nes)])
        clusters[dataset] = df_clusters
    return clusters, ne_map


def update_clusters(data: dict, ne_map: dict):
    for dataset, langs in data.items():
        for lang, docs in langs.items():
            for docId, doc in docs.items():
                doc['content']['calcClId'] = 'xxx'
                for i, row in doc['content'].iterrows():
                    ne_key = f'{lang};{row["docId"]};{row["sentenceId"]};{row["tokenId"]}'
                    if ne_key not in ne_map[dataset]:
                        # if row['calcNER'] != 'O':
                        #     print(f"Missed {ne_key}")
                        continue
                    doc['content'].loc[i, 'calcClId'] = ne_map[dataset][ne_key][0]
                # print('done')
    return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-path', type=str, default=None)
    parser.add_argument('--cluster-path', type=str, default=None)
    return parser.parse_args()


def main():
    global pred_path, cluster_path
    args = parse_args()
    pred_path = args.pred_path if args.pred_path is not None else pred_path
    cluster_path = args.cluster_path if args.cluster_path is not None else cluster_path
    path = pathlib.Path(pred_path)
    if not path.exists() or not path.is_dir():
        raise Exception(f"Path does not exist or is not a directory: `{pred_path}`")
    path = pathlib.Path(cluster_path)
    if not path.exists() or not path.is_dir():
        raise Exception(f"Path does not exist or is not a directory: `{cluster_path}`")

    logger.info("Loading the clusters...")
    clusters, ne_map = load_clusters(cluster_path)

    logger.info("Loading the documents...")
    data = LoadBSNLPDocuments(year='2021', path=f'{pred_path}/predictions/bsnlp/bert-base-multilingual-cased-bsnlp-2021-all-5-epochs').load_predicted()

    logger.info("Merging the cluster data into the prediction data")
    updated = update_clusters(data, ne_map)

    logger.info("Persisting the changes...")
    UpdateBSNLPDocuments(year='2021', path=f'{pred_path}/predictions/bsnlp/bert-base-multilingual-cased-bsnlp-2021-all-5-epochs').update_clustered(updated)

    logger.info("Done.")


if __name__ == '__main__':
    main()
