import json
import pandas as pd
import dedupe

from collections import defaultdict
from itertools import combinations, product


def preprocess(fname: str) -> dict:
    data = pd.read_csv(fname, dtype={'docId': str, 'sentenceId': str, 'tokenId': str, 'lemma': str, 'clID': str})
    data = data.loc[~data["ner"].isin(['O'])]

    data = data.to_dict(orient='records')
    ret = {}
    for entry in data:
        ret[f"{entry['docId']}-{entry['sentenceId']}-{entry['text']}"] = entry
    return ret


def training_examples(data) -> dict:
    positive_examples = defaultdict(list)
    matches = []
    distinct = []

    for key, value in data.items():
        positive_examples[value['clID']].append(value)
    # print(json.dumps(dict(positive_examples), indent=4))

    for key, values in positive_examples.items():
        print(f"{key} ({len(values)}): {values}")
        if len(values) < 2:
            continue
        for comb in combinations(values, 2):
            matches.append(comb)

    clids = positive_examples.keys()
    for comb in combinations(clids, 2):
        # TODO: subsample this
        for items in product(positive_examples[comb[0]], positive_examples[comb[1]]):
            distinct.append(items)

    # print(json.dumps(matches, indent=4))
    # print(json.dumps(distinct, indent=4))

    return {
        'distinct': distinct,
        'match': matches
    }


if __name__ == '__main__':
    print("Dedupe entity matching")
    variables = [
        # docId,sentenceId,tokenId,text,lemma,calcLemma,upos,xpos,ner,clID
        # {"field": "docId", "type": "String"},
        # {"field": "sentenceId", "type": "String"},
        # {"field": "tokenId", "type": "String"},
        {"field": "text", "type": "String"},
        {"field": "calcLemma", "type": "String"},
        {"field": "upos", "type": "String"},
        {"field": "xpos", "type": "String"},
        {"field": "ner", "type": "String"},
    ]
    deduper = dedupe.Dedupe(variables)
    train_data = preprocess("./data/challenge/2021/asia_bibi/merged/sl/sl-1.csv")
    # td = training_examples(train_data)
    test_data = preprocess("./data/challenge/2021/asia_bibi/merged/sl/sl-0.csv")
    td = training_examples(test_data)
    with open('data/deduper/train.json', 'w') as tf:
        # deduper.write_training(tf)
        json.dump(td, tf)

    with open('data/deduper/train.json') as tf:
        deduper.prepare_training(test_data, training_file=tf)
    # dedupe.console_label(deduper)
    deduper.train()

    with open('data/deduper/learned_settings.json', 'wb') as ts:
        deduper.write_settings(ts)

    # clustered = deduper.partition(train_data, 0.5)
    clustered = deduper.partition(train_data, 0.5)
    print('# duplicate sets', len(clustered))
    membership = {}
    for clid, (rec, score) in enumerate(clustered):
        print(f"{clid}: {','.join(rec)}")
        for rid, score in zip(rec, score):
            membership[rid] = {
                "clID": clid,
                "score": str(score)
            }
    print(json.dumps(membership, indent=4))
