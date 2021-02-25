# BSNLP Shared task 2021
*Laboratory for Data Technologies, University of Ljubljana*

Shared task official [website](http://bsnlp.cs.helsinki.fi/shared-task.html)

The guidelines are [here](http://bsnlp.cs.helsinki.fi/System_response_guidelines-1.2.pdf).

## Setup

put the following models in the `data/models/` directory:
- bert-base-multilingual-cased [link](https://huggingface.co/bert-base-multilingual-cased)
- bert-base-multilingual-uncased [link](https://huggingface.co/bert-base-multilingual-uncased)
- cro-slo-eng-bert [link](https://www.clarin.si/repository/xmlui/handle/11356/1330)
- sloberta-1.0 [link](https://www.clarin.si/repository/xmlui/handle/11356/1387)
- sloberta-2.0 [link](https://www.clarin.si/repository/xmlui/handle/11356/1397)

### Running locally
```
pip install -r requirements.txt
```
Run the `./bin/exec-*.sh` scripts.

### Running on a SLURM-enabled cluster
These instructions assume that you will be running on a nVidia-enabled cluster, with [enroot](https://github.com/nvidia/enroot)containers.
```
sbatch ./bin/run-setup.sh
```
Afterwards, run the `./bin/run-*.sh` scripts as required.

Should you need to run the code in a singularity-enabled SLURM cluster, take a look at [this file](./bin/singularity-commands.sh) and how to [run it](./bin/run-singularity.sh).

## Order of execution

1. run [`src/analyze/main.py`](./src/analyze/main.py) ([script](./bin/exec-main.sh)) to get the dataset file structure
2. run [`src/transform/annotate_docs.py`](./src/transform/annotate_docs.py) ([script](./bin/exec-annotate.sh)) to tokenize the dataset and to obtain the lemmas. This will generate the `data/bsnlp/<dataset-name>/merged/<lang>` files
3. run [`src/transform/create_splits.py`](src/transform/create_splits.py)  ([script](./bin/exec-splits.sh)) this will split the dataset into training, validation, and test sets for each language, and store them into `data/bsnlp/<dataset-name>/merged/<lang>/(dev|test|train)_{lang}.csv`. Note: the split is performed on sentences. Each sentence is chosen at random, to preserve the context of all the named entities
4. run [`src/train/crosloeng.py`](./src/train/crosloeng.py) ([script](./bin/run-bert-train.sh)) to train the models.
5. run [`src/eval/model_eval.py`](./src/eval/model_eval.py) ([script](./bin/run-bert-pred.sh)) to generate the predictions of the trained models. Results are stored in `./data/runs/run_<JOB_ID>/`.
6. run [`src/matching/match_dedupe.py`](./src/matching/match_dedupe.py) ([script](./bin/run-dedupe.sh)) to obtain the NE linkage. Results stored in `./data/deduper/runs/run_<JOB_ID>/` (self-created)
7. run (TODO) to merge the results from the entity linking and the NER tasks.
8. run [`src/utils/prepare_output.py`](`./src/utils/prepare_output.py`) ([script](./bin/exec-output.sh)) to generate the output files in BSNLP-compliant format
9. run the evaluation [script](./bin/run-eval.sh) provided by BSNLP organizers to obtain final results. Note: you need to provide a golden standard dataset, see the script on more info.

 Mind the arguments you pass into the scripts, for more details look at the `parse_args` functions in the respective `.py` files, such as [here](./src/train/crosloeng.py).

## Algorithm (open setting):

* INPUT
 * Text documents from web including online media. Each collection about certain event or entity.
* OUTPUT 
 * Recognized entities for each document without indexes. Each entity *lemmatized* and *linked between documents and languages*. Taks: NERC (PER, ORG, LOC, EVT, PRO), name lemmatization, entity matching
 * NOTE: evaluation is case-insensitive, i.e. test data in lower case.

Part-based submission also taken into account.

## Evaluation:

* Relaxed
* Strict
 * Exactly one annotation per entity instance (deduplication) 

## Dataset format

```
Named-entity-mention <TAB> base-form <TAB> category <TAB> cross-lingual ID
```
## Live documentation

[https://drive.google.com/drive/folders/1Zr4dIuEnBmE4yOvSdph7MQc_gnu85ISZ](https://drive.google.com/drive/folders/1Zr4dIuEnBmE4yOvSdph7MQc_gnu85ISZ)
