import argparse
import pandas as pd

from src.utils.load_documents import LoadBSNLPDocuments
from src.utils.update_documents import UpdateBSNLPDocuments
from src.utils.utils import list_dir


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, default='all')
    parser.add_argument('--year', type=str, default='2021')
    parser.add_argument('--run-path', type=str, default=None)
    return parser.parse_args()


def convert_files(
    run_path: str,
    lang: str = 'sl',
    year: str = '2021',
) -> None:
    dirs, _ = list_dir(f'{run_path}/predictions/bsnlp')
    for dir in dirs:
        print(f"Working on {dir}")
        loader = LoadBSNLPDocuments(year=year, lang=lang, path=f'{run_path}/predictions/bsnlp/{dir}')
        updater = UpdateBSNLPDocuments(year=year, lang=lang, path=f'{run_path}/out/{dir}')
        data = loader.load_predicted(folder='clustered')
        # data = loader.load_predicted()
        updater.update_predicted(data)


if __name__ == '__main__':
    args = parser_args()
    print(f'Run path: {args.run_path}')
    print(f'Lang: {args.lang}')
    print(f'Year: {args.year}')
    convert_files(args.run_path, lang=args.lang, year=args.year)

