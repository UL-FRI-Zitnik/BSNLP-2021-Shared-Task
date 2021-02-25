import os
import json
import pandas as pd
import pathlib
import shutil

from src.utils.utils import list_dir
from sklearn.model_selection import train_test_split

# TODO: add different seed option
random_state = 42
TRAIN_SIZE = 0.8


def join_docs(path: str, docs: list) -> pd.DataFrame:
    joined = pd.DataFrame()
    for doc in docs:
        df = pd.read_csv(f'{path}/{doc["merged_fname"]}')
        joined = pd.concat([joined, df])
    return joined


def copy_annotations(
    docs: list,
    path: str,
):
    print(f"Copying annotations to {path}")
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    for doc in docs:
        old_doc_path = pathlib.Path(doc['annotated'])
        ann_name = doc["ann_fname"]
        if ann_name[-4:] != '.out':
            ann_name = f'{ann_name}.out'
        new_doc_path = pathlib.Path(f'{path}/{ann_name}')
        shutil.copy(old_doc_path, new_doc_path)


def join_files(files: list, docs: list) -> list:
    for doc in docs:
        joined = False
        for file in files:
            if file[:-4] == doc['raw_fname'][:-4]:
                doc['merged_fname'] = file
                joined = True
                break
        if not joined:
            print(f"[ERROR] No merged file for {doc}")
    return docs


def create_split(
    dataset_dir: str,
    lang: str,
    docs: list,
    split_path: str,
) -> None:
    path = f"{dataset_dir}/merged/{lang}"
    out_path = f"{dataset_dir}/splits/{lang}/"
    dataset_name = dataset_dir.split('/')[-1]
    print(path)
    _, files = list_dir(path)
    joined = join_files(files, docs)
    train_docs, test_docs = train_test_split(
        joined,
        train_size=TRAIN_SIZE,
        random_state=random_state,
    )
    train_docs, val_docs = train_test_split(
        joined,
        test_size=TRAIN_SIZE * 0.1,
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

    copy_annotations(train_docs, f'{split_path}/train/{dataset_name}/{lang}')
    copy_annotations(val_docs, f'{split_path}/dev/{dataset_name}/{lang}')
    copy_annotations(test_docs, f'{split_path}/test/{dataset_name}/{lang}')


def create_splits(
    datasets: dict,
    split_path: str
) -> None:
    for dataset, langs in datasets.items():
        for lang, docs in langs.items():
            create_split(dataset, lang, docs, split_path)


def main():
    split_path = './data/datasets/bsnlp_splits'
    datasets = json.load(open('./data/results/dataset_pairs.json'))
    create_splits(datasets, split_path)


if __name__ == '__main__':
    main()
