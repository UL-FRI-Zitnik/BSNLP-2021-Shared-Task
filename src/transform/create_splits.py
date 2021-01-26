import os
import json
import pandas as pd

from src.utils.utils import list_dir
from sklearn.model_selection import train_test_split

random_state = 42


def join_docs(path: str, docs: list) -> pd.DataFrame:
    joined = pd.DataFrame()
    for doc in docs:
        df = pd.read_csv(f'{path}/{doc}')
        joined = pd.concat([joined, df])
    return joined


def create_split(dataset_dir: str, lang: str):
    path = f"{dataset_dir}/merged/{lang}"
    out_path = f"{dataset_dir}/splits/{lang}/"
    print(path)
    _, files = list_dir(path)
    train_docs, test_docs = train_test_split(
        files,
        train_size=0.8,
        random_state=random_state,
    )
    train_docs, val_docs = train_test_split(
        train_docs,
        test_size=0.7 * 0.1,
        random_state=random_state,
    )
    # print(len(files), len(train_docs), len(val_docs), len(test_docs))
    train_data = join_docs(path, train_docs)
    val_data = join_docs(path, val_docs)
    test_data = join_docs(path, test_docs)

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    print(f"Saving to: {out_path}")
    train_data.to_csv(f'{out_path}/train_{lang}.csv', index=False)
    val_data.to_csv(f'{out_path}/dev_{lang}.csv', index=False)
    test_data.to_csv(f'{out_path}/test_{lang}.csv', index=False)


def create_splits(datasets: dict):
    for dataset, langs in datasets.items():
        for lang in langs.keys():
            if lang != 'sl':
                continue
            create_split(dataset, lang)



def main():
    datasets = json.load(open('./data/results/dataset_pairs.json'))
    create_splits(datasets)


if __name__ == '__main__':
    main()
