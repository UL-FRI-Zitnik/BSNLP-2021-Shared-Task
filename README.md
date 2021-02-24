# BSNLP Shared task 2021
*Laboratory for Data Technologies, University of Ljubljana*

Shared task official Web site: [link](http://bsnlp.cs.helsinki.fi/shared-task.html)

Guidelines (to be updated): [link](http://bsnlp.cs.helsinki.fi/System_response_guidelines-1.2.pdf)

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

1. 

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
