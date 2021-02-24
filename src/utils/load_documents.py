import json
import pandas as pd
from typing import Callable

from src.utils.utils import list_dir


class LoadDocuments:
    def __init__(self, path):
        self.path = path


class LoadBSNLPDocuments(LoadDocuments):
    def __init__(
        self,
        year: str = 'all',
        lang: str = 'all',
        path: str = './data/datasets/bsnlp',
    ) -> None:
        super(LoadBSNLPDocuments, self).__init__(
            path=path
        )
        datasets = {
            "2017": ["ec", "trump"],
            "2021": ["asia_bibi", "brexit", "nord_stream", "other", "ryanair"],
            "all": ["ec", "trump", "asia_bibi", "brexit", "nord_stream", "other", "ryanair"],
        }
        if year not in datasets:
            raise Exception(f"Invalid subset chosen: {year}")
        self.dirs = datasets[year]
        available_langs = ['bg', 'cs', 'pl', 'ru', 'sl', 'uk']
        if lang in available_langs:
            self.langs = [lang]
        elif lang == 'all':
            self.langs = available_langs
        else:
            raise Exception("Invalid language option.")

    def load(
        self,
        ftype: str,
        fun: Callable  # NOTE: all functions must return `dict` type with `docId` available
    ) -> dict:
        data = {}
        for dataset in self.dirs:
            data[dataset] = {}
            for lang in self.langs:
                data[dataset][lang] = {}
                path = f'{self.path}/{dataset}/{ftype}/{lang}'
                _, files = list_dir(path)
                for fname in files:
                    result = fun(f'{path}/{fname}')
                    result['fname'] = fname
                    data[dataset][lang][result['docId']] = result
        return data

    def load_raw(self) -> dict:
        def raw_loader(fpath: str) -> dict:
            data = {}
            with open(fpath) as f:
                lines = f.readlines()
                data['docId'] = lines[0].strip()
                data['lang'] = lines[1].strip()
                data['created'] = lines[2].strip()
                data['url'] = lines[3].strip()
                data['title'] = lines[4].strip()
                content = ' '.join([line.strip() for line in lines[4:]])
                data['content'] = content
            return data
        return self.load('raw', raw_loader)

    def load_merged(self) -> dict:
        def merged_loader(fpath: str) -> dict:
            df = pd.read_csv(fpath, dtype={'docId': str, 'clID': str}).to_dict(orient='records')
            docId = df[0]['docId']
            return {
                'docId': docId,
                'content': df
            }
        return self.load('merged', merged_loader)

    def load_predicted(self) -> dict:
        def predicted_loader(fpath: str) -> dict:
            df = pd.read_csv(fpath)
            docId = df.iloc[0]['docId']
            return {
                'docId': docId,
                'content': df
            }
        return self.load('predicted', predicted_loader)

    def load_annotated(self):
        def annotated_loader(fpath: str) -> dict:
            docId = open(fpath).readline().strip()
            data = pd.read_csv(
                fpath,
                header=None,
                skiprows=[0],
                delimiter='\t',
                names=['Mention', 'Base', 'Category', 'clID']
            )
            return {
                'docId': docId,
                'content': data.to_dict(orient='records'),
            }
        return self.load('annotated', annotated_loader)


if __name__ == '__main__':
    doc_loader = LoadBSNLPDocuments(lang='sl')
    res = doc_loader.load_annotated()
    print(json.dumps(res, indent=4))
