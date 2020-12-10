# BSNLP Shared task 2021
*Laboratory for Data Technologies, University of Ljubljana*

Shared task official Web site: [http://bsnlp.cs.helsinki.fi/shared-task.html](http://bsnlp.cs.helsinki.fi/shared-task.html)

Guidelines (to be updated): [http://bsnlp.cs.helsinki.fi/Guidelines_20190122.pdf](http://bsnlp.cs.helsinki.fi/Guidelines_20190122.pdf)

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