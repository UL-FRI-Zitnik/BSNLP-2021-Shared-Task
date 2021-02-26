import pandas as pd
import numpy as np
import torch
import random
import os
import sys
import logging
import argparse
import transformers
import pathlib

from datetime import datetime
from tqdm import trange, tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW
from transformers import get_linear_schedule_with_warmup, PreTrainedModel
from keras.preprocessing.sequence import pad_sequences
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
from matplotlib import pyplot as plt
from itertools import product

from src.train.model import Model
from src.utils.load_dataset import LoadSSJ500k, LoadBSNLP, LoadCombined
from src.utils.utils import list_dir

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
)
logger = logging.getLogger('TrainEvalModels')


class BertModel(Model):
    def __init__(
            self,
            tag2code,
            code2tag,
            output_model_path: str,  # this is the output dir
            output_model_fname: str,  # this is the output file name
            tune_entire_model: bool,
            epochs: int = 3,
            max_grad_norm: float = 1.0,
            input_model_path: str = 'data/models/cro-slo-eng-bert',  # this is a directory

    ):
        super().__init__()
        self.input_model_path = input_model_path
        self.output_model_path = output_model_path
        self.output_model_fname = output_model_fname
        logger.info(f"Output model at: {output_model_path}")

        logger.info(f"Tuning entire model: {tune_entire_model}")
        self.tune_entire_model = tune_entire_model

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.input_model_path,
            from_pt=True,
            do_lower_case=False,
            use_fast=False,

        )
        self.MAX_LENGTH = 128  # max input length
        self.BATCH_SIZE = 32  # max input length
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.tag2code, self.code2tag = tag2code, code2tag
        logger.info(f"tags: {self.tag2code.keys()}")
        self.save_weights = False

    def convert_input(self, input_data: pd.DataFrame):
        tokens = []
        tags = []  # NER tags

        if 'docId' not in input_data.columns:
            input_data['docId'] = 'xxx'

        for (_, sentence), data in input_data.groupby(["docId", "sentenceId"]):
            sentence_tokens = []
            sentence_tags = []
            for id, word_row in data.iterrows():
                word_tokens = self.tokenizer.tokenize(str(word_row["text"]))
                sentence_tokens.extend(word_tokens)
                sentence_tags.extend([self.tag2code[word_row["ner"]]] * len(word_tokens))

            sentence_ids = self.tokenizer.convert_tokens_to_ids(sentence_tokens)
            tokens.append(sentence_ids)
            tags.append(sentence_tags)
        # padding is required to spill the sentence tokens in case there are sentences longer than 128 words
        # or to fill in the missing places to 128 (self.MAX_LENGTH)
        tokens = torch.as_tensor(pad_sequences(
            tokens,
            maxlen=self.MAX_LENGTH,
            dtype="long",
            value=0.0,
            truncating="post",
            padding="post"
        )).to(self.device)
        tags = torch.as_tensor(pad_sequences(
            tags,
            maxlen=self.MAX_LENGTH,
            dtype="long",
            value=self.tag2code["PAD"],
            truncating="post",
            padding="post"
        )).to(self.device)
        masks = torch.as_tensor(np.array([[float(token != 0.0) for token in sentence] for sentence in tokens])).to(
            self.device)
        data = TensorDataset(tokens, masks, tags)
        sampler = RandomSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=self.BATCH_SIZE)

    def convert_output(self):
        pass

    def train(
            self,
            data_loaders: dict
    ):
        logger.info(f"Loading the pre-trained model `{self.input_model_path}`...")
        model = AutoModelForTokenClassification.from_pretrained(
            self.input_model_path,
            num_labels=len(self.tag2code),
            label2id=self.tag2code,
            id2label=self.code2tag,
            output_attentions=False,
            output_hidden_states=False
        )

        model = model.to(self.device)
        optimizer, loss = None, None

        for dataset, dataloader in data_loaders.items():
            logger.info(f'Training on `{dataset}`')
            model, optimizer, loss = self.__train(model, train_data=dataloader.train(),
                                                  validation_data=dataloader.dev())

        out_fname = f"{self.output_model_path}/{self.output_model_fname}"
        logger.info(f"Saving the model at: {out_fname}")
        model.save_pretrained(out_fname)
        self.tokenizer.save_pretrained(out_fname)
        logger.info("Done!")

    def __train(
            self,
            model,
            train_data: pd.DataFrame,
            validation_data: pd.DataFrame
    ):
        logger.info("Loading the training data...")
        train_data = self.convert_input(train_data)
        logger.info("Loading the validation data...")
        validation_data = self.convert_input(validation_data)

        if self.tune_entire_model:
            model_parameters = list(model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_parameters = [
                {
                    'params': [p for n, p in model_parameters if not any(nd in n for nd in no_decay)],
                    'weight_decay_rate': 0.01
                },
                {
                    'params': [p for n, p in model_parameters if any(nd in n for nd in no_decay)],
                    'weight_decay_rate': 0.0
                }
            ]
        else:
            model_parameters = list(model.named_parameters())
            optimizer_parameters = [{"params": [p for n, p in model_parameters]}]

        optimizer = AdamW(
            optimizer_parameters,
            lr=3e-5,
            eps=1e-8
        )

        total_steps = len(train_data) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # ensure reproducibility
        # TODO: try out different seed values
        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        training_loss, validation_loss, loss = [], [], None
        logger.info(f"Training the model for {self.epochs} epochs...")
        for _ in trange(self.epochs, desc="Epoch"):
            model.train()
            total_loss = 0
            # train:
            for step, batch in tqdm(enumerate(train_data), desc='Batch'):
                batch_tokens, batch_masks, batch_tags = tuple(t.to(self.device) for t in batch)

                # reset the grads
                model.zero_grad()

                outputs = model(
                    batch_tokens,
                    attention_mask=batch_masks,
                    labels=batch_tags
                )

                loss = outputs[0]
                loss.backward()
                total_loss += loss.item()

                # preventing exploding gradients
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.max_grad_norm)

                # update the parameters
                optimizer.step()

                # update the learning rate (lr)
                scheduler.step()

            avg_epoch_train_loss = total_loss / len(train_data)
            logger.info(f"Avg train loss = {avg_epoch_train_loss:.4f}")
            training_loss.append(avg_epoch_train_loss)

            # validate:
            model.eval()
            val_loss, val_acc, val_f1, val_p, val_r, val_report = self.__test(model, validation_data)
            validation_loss.append(val_loss)
            logger.info(f"Validation loss: {val_loss:.4f}")
            logger.info(f"Validation accuracy: {val_acc:.4f}, P: {val_p:.4f}, R: {val_r:.4f}, F1 score: {val_f1:.4f}")
            logger.info(f"Classification report:\n{val_report}")

        fig, ax = plt.subplots()
        ax.plot(training_loss, label="Traing loss")
        ax.plot(validation_loss, label="Validation loss")
        ax.legend()
        ax.set_title("Model Loss")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epoch")
        fig.savefig(f"{self.output_model_path}/{self.output_model_fname}-loss.png")
        return model, optimizer, loss

    def translate(self, predictions: list, labels: list, tokens) -> (list, list, list):
        translated_predictions, translated_labels, translated_tokens = [], [], []
        for preds, labs, toks in zip(predictions, labels, tokens):
            sentence_predictions, sentence_labels, sentence_tokens = [], [], []
            for p, l, t in zip(preds, labs, toks):
                if l == self.tag2code["PAD"]:
                    continue
                sentence_tokens.append(t)
                sentence_predictions.append(self.code2tag[p])
                sentence_labels.append(self.code2tag[l])
            translated_tokens.append(sentence_tokens)
            translated_predictions.append(sentence_predictions)
            translated_labels.append(sentence_labels)
        return translated_predictions, translated_labels, translated_tokens

    def __test(self, model: PreTrainedModel, data: DataLoader) -> (float, float, float, float, float, str):
        eval_loss = 0.
        eval_steps, eval_examples = 0, 0
        tokens, eval_predictions, eval_labels = [], [], []
        model.eval()
        for batch in tqdm(data):
            batch_tokens, batch_masks, batch_tags = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                outputs = model(
                    batch_tokens,
                    attention_mask=batch_masks,
                    labels=batch_tags
                )
            logits = outputs[1].detach().cpu().numpy()
            label_ids = batch_tags.to('cpu').numpy()
            toks = batch_tokens.to('cpu').numpy()

            eval_loss += outputs[0].mean().item()
            batch_toks = [self.tokenizer.convert_ids_to_tokens(sentence) for sentence in toks]
            tokens.extend(batch_toks)
            eval_predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            eval_labels.extend(label_ids)

            eval_examples += batch_tokens.size(0)
            eval_steps += 1

        eval_loss = eval_loss / eval_steps

        predicted_tags, valid_tags, tokens = self.translate(eval_predictions, eval_labels, tokens)
        for st, sp, sv in zip(tokens, predicted_tags, valid_tags):
            for t, p, v in zip(st, sp, sv):
                logger.info(f"row = {t}, {p}, {v}")

        score_acc = accuracy_score(valid_tags, predicted_tags)
        score_f1 = f1_score(valid_tags, predicted_tags)
        score_p = precision_score(valid_tags, predicted_tags)
        score_r = recall_score(valid_tags, predicted_tags)
        report = classification_report(valid_tags, predicted_tags)

        return eval_loss, score_acc, score_f1, score_p, score_r, report

    def test(self, test_data: pd.DataFrame) -> (float, float, float):
        if not (os.path.exists(self.output_model_path) and os.path.isdir(self.output_model_path)):
            raise Exception(f"A model with the given parameters has not been trained yet,"
                            f" or is not located at `{self.output_model_path}`.")
        models, _ = list_dir(self.output_model_path)
        models = [model_fname for model_fname in models if model_fname.startswith(self.output_model_fname)]
        print("Models:", models)
        if not models:
            raise Exception(f"There are no trained models with the given criteria: `{self.output_model_fname}`")

        logger.info("Loading the testing data...")
        test_data = self.convert_input(test_data)
        avg_acc, avg_f1, avg_p, avg_r, reports = [], [], [], [], []
        for model_fname in models:
            logger.info(f"Loading {model_fname}...")
            model = AutoModelForTokenClassification.from_pretrained(
                f"{self.output_model_path}/{model_fname}",
                num_labels=len(self.tag2code),
                label2id=self.tag2code,
                id2label=self.code2tag,
                output_attentions=False,
                output_hidden_states=False
            )
            model = model.to(self.device)
            _, acc, f1, p, r, report = self.__test(model, test_data)
            avg_acc.append(acc)
            avg_f1.append(f1)
            avg_p.append(p)
            avg_r.append(r)
            logger.info(f"Testing P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}")
            logger.info(f"Testing classification report:\n{report}")
        logger.info(f"Average accuracy: {np.mean(avg_acc):.4f}")
        f1 = np.mean(avg_f1)
        p = np.mean(avg_p)
        r = np.mean(avg_r)
        logger.info(f"Average P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}")
        return p, r, f1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--train-iterations', type=int, default=1)
    parser.add_argument('--train-bundle', type=str, default="slo_misc-only")
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--run-path', type=str, default=None)
    parser.add_argument('--full-finetuning', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    global JOB_ID
    JOB_ID = os.environ['SLURM_JOB_ID'] if 'SLURM_JOB_ID' in os.environ else None
    logger.info(f"Training new NER models")
    logger.info(f"SLURM_JOB_ID = {JOB_ID}")
    logger.info(f"Training: {args.train}")
    logger.info(f"Train iterations: {args.train_iterations}")
    logger.info(f"Train bundle: {args.train_bundle}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Full finetuning: {args.full_finetuning}")
    logger.info(f"Testing: {args.test}")
    logger.info(f"Torch version {torch.__version__}")
    logger.info(f"Transformers version {transformers.__version__}")

    train_bundles = {
        "slo_misc": {
            "models": [
                "cro-slo-eng-bert",
                "bert-base-multilingual-cased",
                "bert-base-multilingual-uncased",
                "sloberta-1.0",
                "sloberta-2.0",
            ],
            "train": {
                "ssj500k-bsnlp2017-iterative": {
                    "ssj500k": LoadSSJ500k(),
                    "bsnlp-2017": LoadBSNLP(lang='sl', year='2017'),
                },
                "ssj500k-bsnlp-2017-combined": {
                    "combined": LoadCombined([LoadSSJ500k(), LoadBSNLP(lang='sl', year='2017')]),
                },
                "ssj500k-bsnlp-2021-iterative": {
                    "ssj500k": LoadSSJ500k(),
                    "bsnlp2021": LoadBSNLP(lang='sl', year='2021'),
                },
                "ssj500k-bsnlp-2021-combined": {
                    "combined": LoadCombined([LoadSSJ500k(), LoadBSNLP(lang='sl', year='2021')]),
                },
                "ssj500k-bsnlp-all-iterative": {
                    "ssj500k": LoadSSJ500k(),
                    "bsnlp2017": LoadBSNLP(lang='sl', year='all'),
                },
                "ssj500k-bsnlp-all-combined": {
                    "combined": LoadCombined([LoadSSJ500k(), LoadBSNLP(lang='sl', year='all')]),
                },
                "ssj500k": {
                    "ssj500k": LoadSSJ500k(),
                },
                "bsnlp-2017": {
                    "bsnlp-2017": LoadBSNLP(lang='sl', year='2017'),
                },
                "bsnlp-2021": {
                    "bsnlp-2021": LoadBSNLP(lang='sl', year='2021'),
                },
                "bsnlp-all": {
                    "bsnlp-all": LoadBSNLP(lang='sl', year='all'),
                },
            },
            "test": {
                "ssj500k": LoadSSJ500k(),
                "bsnlp-2017": LoadBSNLP(lang='sl', year='2017'),
                "bsnlp-2021": LoadBSNLP(lang='sl', year='2021'),
                "bsnlp-all": LoadBSNLP(lang='sl', year='all')
            },
        },
        "slo_misc-only": {
            "models": [
                "cro-slo-eng-bert",
                "bert-base-multilingual-cased",
                "bert-base-multilingual-uncased",
                "sloberta-1.0",
                "sloberta-2.0",
            ],
            "train": {
                "bsnlp-2021": {
                    "bsnlp-2021": LoadBSNLP(lang='sl', year='2021', merge_misc=False, misc_data_only=True),
                }
            },
            "test": {
                "bsnlp-2021": LoadBSNLP(lang='sl', year='2021', merge_misc=False, misc_data_only=True),
            },
        },
        "slo_all": {
            "models": [
                "cro-slo-eng-bert",
                "bert-base-multilingual-cased",
                "bert-base-multilingual-uncased",
                "sloberta-1.0",
                "sloberta-2.0",
            ],
            "train": {
                "bsnlp-2021": {
                    "bsnlp-2021": LoadBSNLP(lang='sl', year='2021', merge_misc=False),
                }
            },
            "test": {
                "bsnlp-2021": LoadBSNLP(lang='sl', year='2021', merge_misc=False),
            }
        },
        "multilang_all": {
            "models": [
                "bert-base-multilingual-cased",
            ],
            "train": {
                'bsnlp-2021-bg': {'bsnlp-2021-bg': LoadBSNLP(lang='bg', year='2021', merge_misc=False)},
                'bsnlp-2021-cs': {'bsnlp-2021-cs': LoadBSNLP(lang='cs', year='2021', merge_misc=False)},
                'bsnlp-2021-pl': {'bsnlp-2021-pl': LoadBSNLP(lang='pl', year='2021', merge_misc=False)},
                'bsnlp-2021-ru': {'bsnlp-2021-ru': LoadBSNLP(lang='ru', year='2021', merge_misc=False)},
                'bsnlp-2021-sl': {'bsnlp-2021-sl': LoadBSNLP(lang='sl', year='2021', merge_misc=False)},
                'bsnlp-2021-uk': {'bsnlp-2021-uk': LoadBSNLP(lang='uk', year='2021', merge_misc=False)},
                'bsnlp-2021-all': {'bsnlp-2021-all': LoadBSNLP(lang='all', year='2021', merge_misc=False)},
            },
            "test": {
                "bsnlp-2021-bg": LoadBSNLP(lang='bg', year='2021', merge_misc=False),
                "bsnlp-2021-cs": LoadBSNLP(lang='cs', year='2021', merge_misc=False),
                "bsnlp-2021-pl": LoadBSNLP(lang='pl', year='2021', merge_misc=False),
                "bsnlp-2021-ru": LoadBSNLP(lang='ru', year='2021', merge_misc=False),
                "bsnlp-2021-sl": LoadBSNLP(lang='sl', year='2021', merge_misc=False),
                "bsnlp-2021-uk": LoadBSNLP(lang='uk', year='2021', merge_misc=False),
                "bsnlp-2021-all": LoadBSNLP(lang='all', year='2021', merge_misc=False),
            }
        }
    }

    chosen_bundle = args.train_bundle
    if chosen_bundle not in train_bundles:
        raise Exception(f"Invalid bundle chosen: {chosen_bundle}")

    bundle = train_bundles[chosen_bundle]
    models = bundle['models']
    train_data = bundle['train']
    test_data = bundle['test']

    if not args.run_path:
        run_time = datetime.now().isoformat()[:-7]  # exclude the ms
        run_path = f'./data/runs/run_{JOB_ID if JOB_ID is not None else run_time}_{chosen_bundle}'
    else:
        run_path = args.run_path
        run_time = run_path.split('/')[-1][4:]

    pathlib.Path(run_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{run_path}/models').mkdir(parents=True, exist_ok=True)
    logger.info(f'Running path: `{run_path}`, run time: `{run_time}`')

    tag2code, code2tag = list(test_data.values())[0].encoding()

    test_f1_scores = []
    for model_name, fine_tuning in product(models, [True, False]):
        logger.info(f"Working on model: `{model_name}`...")
        for train_bundle, loaders in train_data.items():
            bert = BertModel(
                tag2code=tag2code,
                code2tag=code2tag,
                epochs=args.epochs,
                input_model_path=f'./data/models/{model_name}',
                output_model_path=f'{run_path}/models',
                output_model_fname=f'{model_name}-{train_bundle}'
                                   f"{'-finetuned' if fine_tuning else ''}"
                                   f'-{args.epochs}-epochs',
                tune_entire_model=fine_tuning,
            )

            if args.train:
                logger.info(f"Training data bundle: `{train_bundle}`")
                bert.train(loaders)

            if args.test:
                for test_dataset, dataloader in test_data.items():
                    logger.info(f"Testing on `{test_dataset}`")
                    p, r, f1 = bert.test(test_data=dataloader.test())
                    test_f1_scores.append({
                        "model_name": model_name,
                        "fine_tuned": fine_tuning,
                        "train_bundle": train_bundle,
                        "epochs": args.epochs,
                        "test_dataset": test_dataset,
                        "precision_score": p,
                        "recall_score": r,
                        "f1_score": f1
                    })
                    logger.info(f"[{train_bundle}][{test_dataset}] P = {p:.4f}, R = {r:.4f}, F1 = {f1:.4f}")
    if args.test:
        scores = pd.DataFrame(test_f1_scores)
        scores.to_csv(f'{run_path}/training_scores-{run_time}.csv', index=False)
    logger.info(f'Entire training suite is done.')


if __name__ == '__main__':
    main()
