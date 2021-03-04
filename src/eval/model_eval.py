import json
import argparse
import tqdm
import logging
import sys
import pandas as pd
import random

from collections import defaultdict

from src.eval.predict import ExtractPredictions
from src.utils.load_documents import LoadBSNLPDocuments
from src.utils.load_dataset import LoadBSNLP
from src.utils.update_documents import UpdateBSNLPDocuments
from src.utils.utils import list_dir


logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
)
logger = logging.getLogger('TrainEvalModels')

DEBUG = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, default='all')
    parser.add_argument('--year', type=str, default='all')
    parser.add_argument('--merge-misc', action='store_true')
    parser.add_argument('--run-path', type=str, default=None)
    return parser.parse_args()


def group_sentences(document: list) -> dict:
    sentences = defaultdict(lambda: "")
    for token in document:
        sentences[token['sentenceId']] = f"{sentences[token['sentenceId']]} {token['text']}"
    return dict(sentences)

def get_label_dicts(path: str) -> (dict, dict):
    with open(f'{path}/config.json') as f:
        config = json.load(f)
        code2tag = {int(k): v for k, v in config['id2label'].items()}
        return config['label2id'], code2tag


def looper(
    run_path: str,
    clang: str,
    model: str,
    year: str,
    categorize_misc: bool = False,
) -> dict:
    loader = LoadBSNLPDocuments(lang=clang, year=year)    

    model_name = model.split('/')[-1]
    logger.info(f"Predicting for {model_name}")
    model_path = f'{run_path}/models/{model}'
    
    tag2code, code2tag = get_label_dicts(model_path)
    misctag2code, misccode2tag = {}, {}

    logger.info(f"tag2code: {tag2code}")
    logger.info(f"code2tag: {code2tag}")

    misc_model, _ = list_dir(f'{run_path}/misc_models')
    if categorize_misc:
        logger.info(f"Using misc model: {misc_model[0]}")
        misctag2code, misccode2tag = get_label_dicts(f'{run_path}/misc_models/{misc_model[0]}')
        logger.info(f"misctag2code: {misctag2code}")
        logger.info(f"misccode2tag: {misccode2tag}")

    predictor = ExtractPredictions(model_path=model_path, tag2code=tag2code, code2tag=code2tag)
    pred_misc = None if not categorize_misc else ExtractPredictions(model_path=f'./{run_path}/misc_models/{misc_model[0]}', tag2code=misctag2code, code2tag=misccode2tag)

    updater = UpdateBSNLPDocuments(lang=clang, year=year, path=f'{run_path}/predictions/bsnlp/{model_name}')
    predictions = {}
    data = loader.load_merged()
    tdset = tqdm.tqdm(data.items(), desc="Dataset")
    scores = []
    for dataset, langs in tdset:
        tdset.set_description(f'Dataset: {dataset}')
        tlang = tqdm.tqdm(langs.items(), desc="Language")
        predictions[dataset] = {}
        for lang, docs in tlang:
            predictions[dataset][lang] = {}
            tlang.set_description(f'Lang: {tlang}')
            for docId, doc in tqdm.tqdm(docs.items(), desc="Docs"):
                to_pred = pd.DataFrame(doc['content'])
                if categorize_misc:
                    # categorize the PRO and EVT to MISC, as the model only knows about it
                    to_pred.loc[to_pred['ner'].isin(['B-PRO', 'B-EVT']), 'ner'] = f'B-MISC'
                    to_pred.loc[to_pred['ner'].isin(['I-PRO', 'I-EVT']), 'ner'] = f'I-MISC'
                doc_scores, pred_data = predictor.predict(to_pred, tag2code, code2tag)
                doc_scores['id'] = f'{lang};{docId}'
                scores.append(doc_scores)
                if pred_misc is not None and len(pred_data.loc[pred_data['ner'].isin(['B-MISC', 'I-MISC'])]) > 0:
                    misc_data = pd.DataFrame(doc['content'])
                    if len(misc_data.loc[~(misc_data['ner'].isin(['B-MISC', 'I-MISC']))]) > 0:
                        # randomly choose a category for (B|I)-MISC category
                        cat = random.choice(['PRO', 'EVT'])
                        misc_data.loc[(misc_data['ner'] == 'B-MISC'), 'ner'] = f'B-{cat}'
                        misc_data.loc[(misc_data['ner'] == 'I-MISC'), 'ner'] = f'I-{cat}'
                    misc_data.loc[~(misc_data['ner'].isin(['B-PRO', 'B-EVT', 'I-PRO', 'I-EVT'])), 'ner'] = 'O'
                    _, misc_pred = pred_misc.predict(misc_data, misctag2code, misccode2tag)
                    pred_data['ner'] = pd.DataFrame(doc['content'])['ner']
                    # update the entries
                    # update wherever there is misc in the original prediction
                    pred_data.loc[pred_data['calcNER'].isin(['B-MISC', 'I-MISC']), 'calcNER'] = misc_pred.loc[pred_data['calcNER'].isin(['B-MISC', 'I-MISC']), 'calcNER']
                    # update wherever the new predictor made a prediction
                    pred_data.loc[misc_pred['calcNER'].isin(['B-PRO', 'B-EVT', 'I-PRO', 'I-EVT']), 'calcNER'] = misc_pred.loc[misc_pred['calcNER'].isin(['B-PRO', 'B-EVT', 'I-PRO', 'I-EVT']), 'calcNER']
                doc['content'] = pred_data.to_dict(orient='records')
                predictions[dataset][lang][docId] = pred_data.loc[~(pred_data['calcNER'] == 'O')].to_dict(orient='records')
    updater.update_merged(data)
    logger.info(f"Done predicting for {model_name}")
    return {
        'model': model_name,
        'preds': predictions,
    }, scores


def main():
    args = parse_args()
    run_path = args.run_path if args.run_path is not None else "./data/models/"
    lang = args.lang
    year = args.year
    merge_misc = args.merge_misc

    print(f"Run path: {run_path}")
    print(f"Langs: {lang}")
    print(f"Year: {year}")
    print(f"Merge misc: {merge_misc}")

    models, _ = list_dir(f'{run_path}/models')
    logger.info(f"Models to predict: {json.dumps(models, indent=4)}")

    # tmodel = tqdm.tqdm(list(map(lambda x: (run_path, lang, x), models)), desc="Model")
    # predictions = pool.map(looper, tmodel)
    # predictions = list(map(looper, tmodel))
    predictions = []
    doc_scores = {}
    for model in tqdm.tqdm(models, desc="Model"):
        logger.info(f"Model: {model}")
        if 'cro-slo-eng-bert-bsnlp-2021-5-epochs' != model:
            continue
        preds, scores = looper(run_path, lang, model,year, merge_misc)
        predictions.append(preds)
        doc_scores[model]= scores
    # logger.info(predictions)
    
    with open(f'{run_path}/all_predictions.json', 'w') as f:
        json.dump(predictions, f)
    with open(f'{run_path}/all_scores.json', 'w') as f:
        json.dump(predictions, f)
    logger.info("Done.")


if __name__ == '__main__':
    main()
