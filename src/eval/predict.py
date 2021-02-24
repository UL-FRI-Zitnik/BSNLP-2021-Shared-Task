import json
import logging
import sys
import torch
import pandas as pd
import numpy as np

from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import PreTrainedModel, pipeline
from collections import defaultdict
from operator import itemgetter
from tqdm import tqdm

from src.utils.load_dataset import LoadBSNLP


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s'
)
logger = logging.getLogger('MakePrediction')


class ExtractPredictions:
    def __init__(
        self,
        tag2code: dict,
        code2tag: dict,
        model_path: str = f'./data/models/bert-base-multilingual-cased-other',
    ):
        """
            A class to extract all the NE predictions from a given tokens
        :param model_path: path to a HuggingFace-transformers pre-trained model for the NER task, such as BERT Base Multilingual (Un)Cased
        """
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            output_attentions=False,
            output_hidden_states=False,
            num_labels=len(tag2code),
            label2id=tag2code,
            id2label=code2tag,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            from_pt=True,
            do_lower_case=False,
            use_fast=False
        )
        self.BATCH_SIZE = 32
        self.MAX_LENGTH = 128

    def convert_input(
        self,
        input_data: pd.DataFrame,
        tag2code: dict,
    ) -> (DataLoader, list):
        all_ids = []
        ids = []  # sentence ids
        tokens = []  # sentence tokens
        token_ids = []  # converted sentence tokens
        tags = []  # NER tags

        for (doc, sentence), data in input_data.groupby(["docId", "sentenceId"]):
            sentence_tokens = []
            sentence_tags = []
            sentence_ids = []
            for id, word_row in data.iterrows():
                word_tokens = self.tokenizer.tokenize(str(word_row["text"]))
                sentence_tokens.extend(word_tokens)
                sentence_tags.extend([tag2code[word_row["ner"]]] * len(word_tokens))
                token_id_str = f'{doc};{sentence};{word_row["tokenId"]}'
                all_ids.append(token_id_str)
                token_id = len(all_ids) - 1
                sentence_ids.extend([token_id] * len(word_tokens))
            if len(sentence_tokens) != len(sentence_tags) != len(sentence_ids):
                raise Exception("Inconsistent output!")
            ids.append(sentence_ids)
            tokens.append(sentence_tokens)
            sentence_token_ids = self.tokenizer.convert_tokens_to_ids(sentence_tokens)
            token_ids.append(sentence_token_ids)
            tags.append(sentence_tags)
        # padding is required to spill the sentence tokens in case there are sentences longer than 128 words
        # or to fill in the missing places to 128 (self.MAX_LENGTH)
        ids = torch.as_tensor(pad_sequences(
            ids,
            maxlen=self.MAX_LENGTH,
            dtype="long",
            value=-1,
            truncating="post",
            padding="post"
        )).to(self.device)
        token_ids = torch.as_tensor(pad_sequences(
            token_ids,
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
            value=tag2code["PAD"],
            truncating="post",
            padding="post"
        )).to(self.device)
        masks = torch.as_tensor(np.array([[float(token != 0.0) for token in sentence] for sentence in token_ids])).to(self.device)
        data = TensorDataset(ids, token_ids, masks, tags)
        sampler = RandomSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=self.BATCH_SIZE), all_ids

    def translate(
        self,
        predictions: list,
        labels: list,
        tokens: list,
        sent_ids: list,
        tag2code: dict,
        code2tag: dict,
        all_ids: list
    ) -> (list, list, list, list):
        translated_predictions, translated_labels, translated_tokens, translated_sentences = [], [], [], []
        for preds, labs, toks, ids in zip(predictions, labels, tokens, sent_ids):
            sentence_predictions, sentence_labels, sentence_tokens, sentence_ids = [], [], [], []
            for p, l, t, i in zip(preds, labs, toks, ids):
                if l == tag2code["PAD"]:
                    continue
                if p == tag2code["PAD"]:
                    logger.info(f"PREDICTED `PAD`! {p}, {l}, {t}, {i}")
                    continue
                sentence_tokens.append(t)
                sentence_predictions.append(code2tag[p])
                sentence_labels.append(code2tag[l])
                sentence_ids.append(all_ids[i])
            translated_tokens.append(sentence_tokens)
            translated_predictions.append(sentence_predictions)
            translated_labels.append(sentence_labels)
            translated_sentences.append(sentence_ids)
        return translated_predictions, translated_labels, translated_tokens, translated_sentences

    def test(
        self,
        data: DataLoader,
        all_ids: list,
        tag2code: dict,
        code2tag: dict,
    ) -> (dict, pd.DataFrame):
        eval_loss = 0.
        eval_steps, eval_examples = 0, 0
        eval_ids, eval_tokens, eval_predictions, eval_labels = [], [], [], []
        self.model.eval()
        for batch in data:
            batch_ids, batch_tokens, batch_masks, batch_tags = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                outputs = self.model(
                    batch_tokens,
                    attention_mask=batch_masks,
                    labels=batch_tags
                )
            logits = outputs[1].detach().cpu().numpy()
            label_ids = batch_tags.to('cpu').numpy()
            toks = batch_tokens.to('cpu').numpy()
            sentence_ids = batch_ids.to('cpu').numpy()

            eval_loss += outputs[0].mean().item()
            toks = [self.tokenizer.convert_ids_to_tokens(sentence) for sentence in toks]
            eval_tokens.extend(toks)
            eval_predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            eval_labels.extend(label_ids)
            eval_ids.extend(sentence_ids)

            eval_examples += batch_tokens.size(0)
            eval_steps += 1
        eval_loss = eval_loss / eval_steps
        flatten = lambda x: [j for i in x for j in i]

        predicted_tags, valid_tags, tokens, sentence_ids = self.translate(eval_predictions, eval_labels, eval_tokens, eval_ids, tag2code, code2tag, all_ids)

        # for st, sp, sv, vi in zip(tokens, predicted_tags, valid_tags, sentence_ids):
        #     for t, p, v, i in zip(st, sp, sv, vi):
        #         logger.info(f"row = {t}, {p}, {v}, {i}")

        predicted_data = pd.DataFrame(data={
            'sentence_id': flatten(sentence_ids),
            'tokens': flatten(tokens),
            'predicted_tag': flatten(predicted_tags),
            'valid_tag': flatten(valid_tags),
        })

        if len([tag for sent in valid_tags for tag in sent if tag[:2] in ['B-', 'I-']]) == 0:
            valid_tags.append(["O"])
            predicted_tags.append(["B-ORG"])


        scores = {
            "loss": eval_loss,
            "acc": accuracy_score(valid_tags, predicted_tags),
            "f1": f1_score(valid_tags, predicted_tags),
            "p": precision_score(valid_tags, predicted_tags),
            "r": recall_score(valid_tags, predicted_tags),
            "report": classification_report(valid_tags, predicted_tags),
        }

        return scores, predicted_data

    def __merge_data(self,
        data: pd.DataFrame,
        pred_data: pd.DataFrame,
    ) -> pd.DataFrame:
        data['calcNER'] = ''
        for sent_id, sent_data in pred_data.groupby('sentence_id'):
            ids = sent_id.split(';')
            did = ids[0]
            sid = int(ids[1])
            tid = int(ids[2])
            max_cat = max(sent_data['predicted_tag'].value_counts().to_dict().items(), key=itemgetter(1))[0]
            data.loc[(data['docId'] == did) & (data['sentenceId'] == sid) & (data['tokenId'] == tid), 'calcNER'] = max_cat
        return data

    def predict(self,
        data: pd.DataFrame,
        tag2code: dict,
        code2tag: dict,
    ) -> (dict, pd.DataFrame):
        in_data, ids = self.convert_input(data, tag2code)
        scores, pred_data = self.test(in_data, ids, tag2code, code2tag)
        merged = self.__merge_data(data, pred_data)
        return scores, merged


if __name__ == '__main__':
    # model_path = f'./data/models/bert-base-multilingual-cased-other'
    model_path = './data/runs/run_2021-02-17T11:42:19_slo-models/models/sloberta-1.0-bsnlp-2021-5-epochs'
    tag2code, code2tag = LoadBSNLP(lang='sl', year='2021', merge_misc=False).encoding()
    logger.info(f'{tag2code}')
    logger.info(f'{code2tag}')
    loader = LoadBSNLP(lang="sl", year='2021', merge_misc=False)
    predictor = ExtractPredictions(model_path)
    data = loader.test()
    scores, pred_data = predictor.predict(data, tag2code, code2tag)
    logger.info(f'{json.dumps(scores, indent=4)}')
    logger.info(f'\n{scores["report"]}')
    logger.info(f'\n{pred_data}')
