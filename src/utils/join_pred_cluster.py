import sys
import json
import glob
import pathlib
import logging
import argparse
import pandas as pd

from src.utils.load_dataset import LoadBSNLP

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
) -> list:
    clusters = []
    for dataset in LoadBSNLP.datasets:
        for fname in glob.glob(f'{path}/{dataset}/cluster-*.json')
            print(fname)
        break
    return []


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
    path = pathlib.Path(run_path)
    if not path.exists() or not path.is_dir():
        raise Exception(f"Path does not exist or is not a directory: `{run_path}`")
    path = pathlib.Path(cluster_path)
    if not path.exists() or not path.is_dir():
        raise Exception(f"Path does not exist or is not a directory: `{cluster_path}`")
    clusters = load_clusters()


if __name__ == '__main__':
    main()