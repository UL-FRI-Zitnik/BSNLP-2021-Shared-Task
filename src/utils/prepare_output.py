import argparse
import pandas as pd

from src.utils.load_documents import LoadBSNLPDocuments
from src.utils.update_documents import UpdateBSNLPDocuments
from src.utils.utils import list_dir


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, default='all')
    parser.add_argument('--run-path', type=str, default=None)
    return parser.parse_args()


def convert_files(
    run_path: str,
    lang: str = 'sl'
) -> None:
    dirs, _ = list_dir(f'{run_path}/predictions/bsnlp')
    for dir in dirs:
        print(f"Working on {dir}")
        loader = LoadBSNLPDocuments(lang=lang, path=f'{run_path}/predictions/bsnlp/{dir}')
        updater = UpdateBSNLPDocuments(lang=lang, path=f'{run_path}/out/{dir}')
        data = loader.load_predicted()
        updater.update_predicted(data)


if __name__ == '__main__':
    args = parser_args()
    print(args.run_path)
    convert_files(args.run_path, lang=args.lang)

