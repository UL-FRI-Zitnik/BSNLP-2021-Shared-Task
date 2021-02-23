import os
import json
import pandas as pd
import sys
import logging

from collections import defaultdict


logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
)
logger = logging.getLogger('main')


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


def list_datasets(datasets: list) -> dict:
    dataset_files = {}
    for dataset in datasets:
        dataset_files[dataset] = {}
        languages_raw = sorted(os.listdir(f'{dataset}/raw'))
        languages_ann = sorted(os.listdir(f'{dataset}/annotated'))
        for lang_id, lang in enumerate(languages_raw):
            base_raw = f'{dataset}/raw/{lang}'
            base_ann = f'{dataset}/annotated/{languages_ann[lang_id]}'
            for r, a in zip(sorted(os.listdir(base_raw)),  sorted(os.listdir(base_ann))):
                digits_r = ''.join([d for d in r if d.isdigit()])
                digits_a = ''.join([d for d in r if d.isdigit()])
                if digits_a != digits_r:
                    raise Exception(f'NO MATCH:\n{base_raw}/{r}\n{base_ann}/{a}')
            dataset_files[dataset][languages_ann[lang_id]] = [{'raw': f'{base_raw}/{r}', 'annotated': f'{base_ann}/{a}', 'raw_fname': r, 'raw_aname': a} for r, a in zip(sorted(os.listdir(base_raw)),  sorted(os.listdir(base_ann)))]
    return dataset_files


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


def raw_doc_info(fname: str) -> dict:
    file_info = {}
    with open(fname,  encoding='utf-8-sig') as f:
        lines = f.readlines()
        file_info['id'] = lines[0].strip()
        file_info['lang'] = lines[1].strip()
        file_info['created'] = lines[2].strip()
        file_info['url'] = lines[3].strip()
        file_info['title'] = lines[4].strip()
        content = ' '.join(lines[5:]).strip()
        file_info['contentLength'] = len(content)
        file_info['numWords'] = len(content.split(' '))
    return file_info


def ann_doc_info(fname: str) -> dict:
    file_info = {}
    ne_categories = ['PER', 'ORG', 'LOC', 'EVT', 'PRO']
    with open(fname, encoding='utf-8-sig') as f:
        lines = f.readlines()
        file_info['id'] = lines[0].strip()
        df = pd.read_csv(fname, names=['Mention', 'Base', 'Category', 'clID'], skiprows=[0], sep='\t')
        file_info['NEcount'] = len(df.index)
        cat_counts = df['Category'].value_counts()
        for cat in ne_categories:
            file_info[cat] = cat_counts[cat] if cat in cat_counts else 0
        file_info['UniqueCLIDs'] = len(df['clID'].unique())
    return file_info


def get_doc_info(stats: dict) -> dict:
    dataset_raw = []
    dataset_ann = []
    for dataset, data in stats.items():
        for lang, files in data['dirs']['raw'].items():
            for file in files:
                info = raw_doc_info(file)
                info['dataset_dir'] = dataset
                info['lang'] = lang
                info['fpath'] = file
                dataset_raw.append(info)
        for lang, files in data['dirs']['annotated'].items():
            for file in files:
                info = ann_doc_info(file)
                info['dataset_dir'] = dataset
                info['lang'] = lang
                info['fpath'] = file
                dataset_ann.append(info)
    raw_df = pd.DataFrame(dataset_raw)
    raw_df.to_csv("./data/results/file_raw_stats.csv")

    ann_df = pd.DataFrame(dataset_ann)
    ann_df.to_csv("./data/results/file_ne_stats.csv")

    return {
        "raw": raw_df,
        "ann": ann_df,
    }


if __name__ == '__main__':
    # dirs, files, packed = list_all_files("./data/challenge")
    # print(tf'Dirs = {json.dumps(dirs, indent=4)}')
    # print(tf'Files = {json.dumps(files, indent=4)}')
    # print(tf'Packed = {json.dumps(packed, indent=4)}')
    # with open('./data/results/file_dump.json', 'w') as tf:
    #     json.dump(packed, tf, indent=4)

    # aggregated = aggregate_nes(packed)
    # with open('./data/results/aggregated.json', 'w') as tf:
    #     json.dump(aggregated, tf)

    # doc_infos = get_doc_info(packed)
    # raw_df = doc_infos['raw']
    # ann_df = doc_infos['ann']

    datasets = [
        './data/bsnlp/ec',
        './data/bsnlp/trump',
        # 2019 data is updated for the 2021 challenge, so these are obsolete
        # './data/bsnlp/sample',
        # './data/bsnlp/training',
        # './data/bsnlp/nord_stream',
        # './data/bsnlp/ryanair',
        './data/bsnlp/asia_bibi',
        './data/bsnlp/brexit',
        './data/bsnlp/nord_stream',
        './data/bsnlp/other',
        './data/bsnlp/ryanair',
    ]
    dataset_files = list_datasets(datasets)
    logger.info('Done.')
    # logger.info(json.dumps(dataset_files, indent=4))
    with open('./data/results/dataset_pairs.json', 'w') as f:
        json.dump(dataset_files, f, indent=4)
