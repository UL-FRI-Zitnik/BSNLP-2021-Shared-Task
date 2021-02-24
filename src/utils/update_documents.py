import pandas as pd

from typing import Callable
from pathlib import Path


class UpdateDocuments:
    def __init__(self, path):
        self.path = path


class UpdateBSNLPDocuments(UpdateDocuments):
    def __init__(
        self,
        year: str = 'all',
        lang: str = 'all',
        path: str = './data/datasets/bsnlp',
    ) -> None:
        super(UpdateBSNLPDocuments, self).__init__(
            path=path
        )
        datasets = {
            "2017": ["ec", "trump"],
            "2021": ["asia_bibi", "brexit", "nord_stream", "other", "ryanair"],
            "all":  ["ec", "trump", "asia_bibi", "brexit", "nord_stream", "other", "ryanair"],
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

    def __update(
        self,
        ftype: str,
        data: dict,
        fun: Callable
    ) -> None:
        for dataset, langs in data.items():
            if dataset not in self.dirs:
                raise Exception(f"Unrecognized dataset: {dataset}")
            for lang, documents in langs.items():
                if lang not in self.langs:
                    raise Exception(f"Unrecognized language: {lang}")
                path = f'{self.path}/{dataset}/{ftype}/{lang}'
                Path(path).mkdir(parents=True, exist_ok=True)
                for docId, content in documents.items():
                    fun(f'{path}/{content["fname"]}', content)

    def update_merged(self, new_data) -> None:
        def update_merged(fpath: str, doc: dict) -> None:
            df = pd.DataFrame(doc['content'])
            df.to_csv(fpath, index=False)
        self.__update('predicted', new_data, update_merged)

    def __merge_records(
        self,
        nes: pd.DataFrame
    ) -> pd.DataFrame:
        """
            Merges the NEs in the form of the expected output
        :param nes:
        :return:
        """
        nes = nes.to_dict(orient='records')
        merged = []
        for i, ne in enumerate(nes):
            if ne['calcNER'].startswith('I-'):
                continue
            j = i + 1
            while j < len(nes) and not nes[j]['calcNER'].startswith('B-'):
                ne['text'] = f'{ne["text"]} {nes[j]["text"]}'
                ne['calcLemma'] = f'{ne["calcLemma"]} {nes[j]["calcLemma"]}'
                j += 1
            ne['calcNER'] = ne['calcNER'][2:]
            merged.append(ne)
        return pd.DataFrame(merged)

    def update_predicted(self, new_data) -> None:
        def update_predicted(fpath: str, doc: dict) -> None:
            df = doc['content']
            if 'calcLemma' not in df.columns:
                print(f"MISSING LEMMA: `{fpath}`")
                df['calcLemma'] = 'xxx'
            df['calcClId'] = 'xxx'
            if 'calcNer' in df.columns:
                df = df.rename(columns={'calcNer': 'calcNER'})
            df = df[['text', 'calcLemma', 'calcNER', 'calcClId']]
            if len(df.loc[df['calcNER'].isna()]) > 0:
                df.loc[df['calcNER'].isna(), 'calcNER'] = 'O'
            df = df.loc[~df['calcNER'].isin(['O'])]
            df = self.__merge_records(df)
            df = df.drop_duplicates(subset=['text'])
            with open(f'{fpath}.out', 'w') as f:
                f.write(f'{doc["docId"]}\n')
                df.to_csv(f,  sep='\t', header=False, index=False)
        self.__update('', new_data, update_predicted)
