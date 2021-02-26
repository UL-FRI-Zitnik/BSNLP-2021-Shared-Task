import os
import sys
import logging
import pandas as pd

from datetime import datetime

from src.train.crosloeng import BertModel
from src.utils.load_dataset import LoadBSNLP

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
)
logger = logging.getLogger('TrainL1OStrategy')

run_time = datetime.now().isoformat()[:-7]  # exclude the ms
JOB_ID = os.environ['SLURM_JOB_ID'] if 'SLURM_JOB_ID' in os.environ else None
run_path = f'./data/runs/run_l1o_{JOB_ID if JOB_ID is not None else run_time}'


def main():
    epochs = 5
    fine_tuning = True
    model_name = 'bert-base-multilingual-cased'
    test_scores = []
    for test_dataset in LoadBSNLP.datasets['2021']:
        train_bundle = f'bsnlp-exclude-{test_dataset}'
        train_datasets = {
            train_bundle: LoadBSNLP(
                        lang='all',
                        year='2021',
                        merge_misc=False,
                        misc_data_only=True,
                        exclude=test_dataset
                    )
        }
        test_dataset = LoadBSNLP(
            lang='all',
            year='2021',
            data_set=test_dataset,
            merge_misc=False,
            misc_data_only=True,
        )
        tag2code, code2tag = test_dataset.encoding()
        bert = BertModel(
            tag2code=tag2code,
            code2tag=code2tag,
            epochs=epochs,
            input_model_path=f'./data/models/{model_name}',
            output_model_path=f'{run_path}/models',
            output_model_fname=f'{model_name}-{train_bundle}'
                               f"{'-finetuned' if fine_tuning else ''}"
                               f'-{epochs}-epochs',
            tune_entire_model=fine_tuning,
            use_test=True,
        )
        logger.info(f"Training data bundle: `{train_bundle}`")
        bert.train(train_datasets)
        logger.info(f"Testing on `{test_dataset}`")
        p, r, f1 = bert.test(test_data=test_dataset.load_all())
        test_scores.append({
            "model_name": model_name,
            "fine_tuned": fine_tuning,
            "train_bundle": train_bundle,
            "epochs": epochs,
            "test_dataset": test_dataset,
            "precision_score": p,
            "recall_score": r,
            "f1_score": f1
        })
        logger.info(f"[{train_bundle}][{test_dataset}] P = {p:.4f}, R = {r:.4f}, F1 = {f1:.4f}")
    scores = pd.DataFrame(test_scores)
    scores.to_csv(f'{run_path}/training_scores-L1O-{JOB_ID}.csv', index=False)


if __name__ == '__main__':
    main()
