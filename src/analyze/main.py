import os
import json
import pandas as pd

from collections import defaultdict


def list_all_files(dirpath: str) -> (list, list, dict):
    files, dirs = [], []
    stats = {
        'test': {
            "numFiles": 0,
            "dirs": {
                "annotated": defaultdict(list),
                "raw": defaultdict(list),
            }
        },
        'sample': {
            "numFiles": 0,
            "dirs": {
                "annotated": defaultdict(list),
                "raw": defaultdict(list),
            }
        },
        'train': {
            "numFiles": 0,
            "dirs": {
                "annotated": defaultdict(list),
                "raw": defaultdict(list),
            }
        },
    }
    for dpath, dnames, fnames in os.walk(dirpath,):
        whole_path_dirs = [f'{dpath}/{dname}' for dname in dnames]
        whole_path_files = [f'{dpath}/{fname}' for fname in fnames]
        dirs.extend(whole_path_dirs)
        files.extend(whole_path_files)
        if fnames:
            if 'test' in dpath:
                stats['test']['numFiles'] += len(whole_path_files)
                if "annotated" in dpath:
                    stats['test']['dirs']['annotated'][dpath[-2:].lower()].extend(whole_path_files)
                else:
                    stats['test']['dirs']['raw'][dpath[-2:].lower()].extend(whole_path_files)

            elif 'sample' in dpath:
                stats['sample']['numFiles'] += len(whole_path_files)
                if "annotated" in dpath:
                    stats['sample']['dirs']['annotated'][dpath[-2:].lower()].extend(whole_path_files)
                else:
                    stats['sample']['dirs']['raw'][dpath[-2:].lower()].extend(whole_path_files)
            else:
                stats['train']['numFiles'] += len(whole_path_files)
                if "annotated" in dpath:
                    stats['train']['dirs']['annotated'][dpath[-2:].lower()].extend(whole_path_files)
                else:
                    stats['train']['dirs']['raw'][dpath[-2:].lower()].extend(whole_path_files)
    return sorted(dirs), sorted(files), stats


def aggregate_nes(stats: dict) -> dict:
    ne_stats = {}
    atts = ['Mention', 'Base', 'Category', 'clID']
    all_data = {att: pd.DataFrame() for att in atts}
    for dataset, data in stats.items():
        ne_stats[dataset] = {}
        for lang, files in data['dirs']['annotated'].items():
            ne_stats[dataset][lang] = {}
            lang_data = pd.DataFrame()
            for file in files:
                file_nes = pd.read_csv(file, header=None, skiprows=[0], delimiter='\t', names=['Mention', 'Base', 'Category', 'clID'])
                lang_data = pd.concat([lang_data, file_nes], ignore_index=True)
            for att in atts:
                counts = pd.DataFrame(lang_data[att].value_counts())
                ne_stats[dataset][lang][att] = counts.to_json()
                counts.reset_index(inplace=True)
                counts = counts.rename(columns={'index': att, att:'Count'})
                all_data[att] = pd.concat([all_data[att], counts], ignore_index=True)
                counts.to_csv(f'./data/stats/{dataset}-{lang}-{att}.csv', index=False)
        for att in atts:
            counts = all_data[att].groupby([att]).agg(['sum'])
            counts.reset_index(inplace=True)
            counts.columns = [att, 'Count']
            counts.to_csv(f'./data/stats/{dataset}-{att}.csv', index=False)
    return ne_stats


def get_doc_len(stats: dict) -> dict:
    
    return {}


if __name__ == '__main__':
    dirs, files, packed = list_all_files("./data/challenge")
    # print(f'Dirs = {json.dumps(dirs, indent=4)}')
    # print(f'Files = {json.dumps(files, indent=4)}')
    # print(f'Packed = {json.dumps(packed, indent=4)}')
    with open ('./data/results/file_dump.json', 'w') as f:
        json.dump(packed, f, indent=4)

    aggregate_nes(packed)
    with open ('./data/results/.json', 'w') as f:
        json.dump(packed, f, indent=4)

