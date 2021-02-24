import pandas as pd
import pyconll
import numpy as np

from sklearn.model_selection import train_test_split
from src.utils.utils import list_dir
from typing import Iterable

# pd.set_option('display.max_rows', None)  # only for debugging purposes


class LoadDataset:
    def __init__(self, base_fname: str, format: str, print_debug: bool = False):
        self.base_fname = base_fname
        self.data_format = format
        self.print_debug = print_debug

    def load(self, dset: str) -> pd.DataFrame:
        return pd.DataFrame()

    def train(self) -> pd.DataFrame:
        return pd.DataFrame()

    def dev(self) -> pd.DataFrame:
        """
            This is the validation data
        """
        return pd.DataFrame()

    def test(self) -> pd.DataFrame:
        return pd.DataFrame()

    def encoding(self) -> (dict, dict):
        data = self.train()
        possible_tags = np.append(data["ner"].unique(), ["PAD"])
        tag2code = {tag: code for code, tag in enumerate(possible_tags)}
        code2tag = {val: key for key, val in tag2code.items()}
        return tag2code, code2tag


class LoadSSJ500k(LoadDataset):
    def __init__(self):
        super().__init__(
            "data/datasets/ssj500k/",
            "conll"
        )

    def load(self, dset: str) -> pd.DataFrame:
        raw_data = pyconll.load_from_file(f"{self.base_fname}{dset}_ner.conllu")
        data = []
        for id, sentence in enumerate(raw_data):
            for word in sentence:
                if word.upos == 'PROPN':  # check if the token is a NER
                    annotation = list(word.misc.keys())[0]
                    data.append({"word": word.form, "sentence": id, "ner": annotation.upper()})
                    # NOTE: we cannot use the just <TYPE> annotation without `B-` (begin) or `I-` (inside) `<TYPE>`
                    # because we would not be compliant with the CoNLL format
                    # annotation = annotation if annotation != "O" else "B-O"
                    # data.append({"word": word.form, "sentence": id, "ner": annotation.split("-")[1].upper()})
                else:
                    data.append({"word": word.form, "sentence": id, "ner": "O"})
        return pd.DataFrame(data)

    def train(self) -> pd.DataFrame:
        return self.load('train')

    def dev(self) -> pd.DataFrame:
        return self.load('dev')

    def test(self) -> pd.DataFrame:
        return self.load('test')


class LoadBSNLP(LoadDataset):
    available_langs = ['bg', 'cs', 'pl', 'ru', 'sl', 'uk']
    datasets = {
        "2017": ["ec", "trump"],
        "2021": ["asia_bibi", "brexit", "nord_stream", "other", "ryanair"],
        "all": ["ec", "trump", "asia_bibi", "brexit", "nord_stream", "other", "ryanair"],
    }

    def __init__(
        self,
        lang: str = 'all',
        year: str = 'all',
        data_set: str = 'all',
        merge_misc: bool = True,
        misc_data_only: bool = False,
        print_debug: bool = False
    ):
        super().__init__(
            "data/datasets/bsnlp",
            "csv",
            print_debug=print_debug,
        )
        # assert year
        if year not in self.datasets:
            raise Exception(f"Invalid year chosen: {year}")

        # assert dataset
        if data_set in self.datasets[year]:
            self.data_set = [data_set]
        elif data_set == 'all':
            self.data_set = self.datasets[year]
        else:
            raise Exception(f"Invalid dataset chosen: {data_set}")

        # assert language
        if lang in self.available_langs:
            self.langs = [lang]
        elif lang == 'all':
            self.langs = self.available_langs
        else:
            raise Exception(f"Invalid language option: {lang}")

        self.random_state = 42
        self.merge_misc = merge_misc
        if merge_misc and misc_data_only:
            print("WARNING: weird combination? merge misc and misc data only?")
        self.misc_data_only = misc_data_only

    def load(self, dset: str) -> pd.DataFrame:
        dirs, _ = list_dir(self.base_fname)
        data = pd.DataFrame()
        for dataset in dirs:
            if dataset not in self.data_set:
                continue
            for lang in self.langs:
                fname = f"{self.base_fname}/{dataset}/splits/{lang}/{dset}_{lang}.csv"
                try:
                    df = pd.read_csv(f"{fname}")
                except:
                    if self.print_debug: print(f"[{dataset}] skipping {lang}.")
                    continue
                df['sentenceId'] = df['docId'].astype(str) + ';' + df['sentenceId'].astype('str') # + '-' + df['tokenId'].astype(str)
                if self.merge_misc:
                    df['ner'] = df['ner'].map(lambda x: x.replace("PRO", "MISC").replace("EVT", "MISC"))
                if self.misc_data_only:
                    df['ner'] = df['ner'].map(lambda x: "O" if x[2:] in ["PER", "LOC", "ORG"] else x)
                data = pd.concat([data, df])
        return data

    def train(self) -> pd.DataFrame:
        return self.load('train')

    def dev(self) -> pd.DataFrame:
        """
            This is the validation data
        """
        return self.load('dev')

    def test(self) -> pd.DataFrame:
        return self.load('test')


class LoadCombined(LoadDataset):
    def __init__(self, loaders: list):
        super().__init__(
            f"combined_datasets:{','.join([l.base_fname for l in loaders])}",
            "csv"
        )
        self.random_state = 42
        self.loaders = loaders

    def load(self, set: str) -> pd.DataFrame:
        return pd.DataFrame()

    def train(self) -> pd.DataFrame:
        data = pd.DataFrame()
        for loader in self.loaders:
            loader_data = loader.train()
            data = pd.concat([data, loader_data])
        return data

    def dev(self) -> pd.DataFrame:
        data = pd.DataFrame()
        for loader in self.loaders:
            loader_data = loader.dev()
            data = pd.concat([data, loader_data])
        return data

    def test(self) -> pd.DataFrame:
        data = pd.DataFrame()
        for loader in self.loaders:
            loader_data = loader.test()
            data = pd.concat([data, loader_data])
        return data


if __name__ == '__main__':
    loader = LoadBSNLP(lang="all", year='2021', merge_misc=False)
    # loader = LoadSSJ500k()
    # loader = LoadCombined([LoadBSNLP("sl"), LoadSSJ500k()])
    tag2code, code2tag = loader.encoding()
    print(f"tag2code: {tag2code}")
    print(f"code2tag: {code2tag}")

    train_data = loader.train()
    # print(train_data.head(10))
    print(f"Train data: {train_data.shape[0]}, NERs: {train_data.loc[train_data['ner'] != 'O'].shape[0]}")
    print(train_data['ner'].value_counts())
    print(train_data.value_counts())
    # print(train_data)
    
    dev_data = loader.dev()
    print(f"Validation data: {dev_data.shape[0]}, NERs: {dev_data.loc[dev_data['ner'] != 'O'].shape[0]}")
    print(dev_data['ner'].value_counts())
    
    test_data = loader.test()
    print(f"Test data: {test_data.shape[0]}, NERs: {test_data.loc[test_data['ner'] != 'O'].shape[0]}")
    print(test_data['ner'].value_counts())
