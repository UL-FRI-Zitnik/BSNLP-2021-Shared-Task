====================================
BSNLP-NER 2019 shared task evaluator
====================================

The package contains a system response validation script and an evaluation script that allows to 
compare the performance of various system runs.

1. VALIDATION
=============

Usage: java -cp bsnlp-ner-evaluator-19.0.4.jar sigslav.ConsistencyCheck [path-to-files] [out-file]

- path-to-file - a path to a directory containing the files to be validated (e.g. data/golden/trump/cs)
- out-file - an output file, which contains the validation info


2. EVALUATION
=============

Usage: java -cp bsnlp-ner-evaluator-19.0.4.jar sigslav.BNEvaluator [path-to-data] [path-to-reports] [path-to-error-log] [path-to-summaries]

- path-to-data - data with gold annotations and system outputs (e.g. data)
- path-to-reports - en empty directory, where per-system evaluation reports will be saved (e.g. reports)
- path-to-error-log - en empty directory, where the per-system per-evaluation metric error reports will be saved (e.g. error-logs)
- path-to-summaries - en empty directory, where the system results' summaries will be saved (a .csv file) (e.g. summaries)

Example: java -cp bsnlp-ner-evaluator-19.0.4.jar sigslav.BNEvaluator data reports error-logs summaries

Data should be organized as follows:

data
- golden
- - trump
- - - cs
- - - - file_103.txt
- - - - file_104.txt
- - - - ...
- - - sk
- - - ru
- - - ...
- - eu
- system-1
- - trump
- - - cs
- - - sk
- - - ru
- - - ...
- - eu
- system-2
...

The document IDs are taken from the file (the first line). The file names are not important.

Results:

1) named entity recognition

1a)  Relaxed evaluation, partial match: an entity mentioned in a given document is
considered to be extracted correctly if the system response includes at
least one annotation of a named mention of this entity (regardless whether
the extracted mention is base form).
Even partial match counts.

1b)  Relaxed evaluation, exact match: an entity mentioned in a given document is
considered to be extracted correctly if the system response includes at
least one annotation of a named mention of this entity (regardless whether
the extracted mention is base form).
The full string have to be matched.

1c) Strict evaluation: the system response should include exactly one
annotation for each unique form of a named mention of an entity that is
referred to in a given document, i.e., capturing and listing all variants
of an entity is required. Partial matches are errors.

2) Name normalisation

Taking all mentions, but only those that need to be normalized on both sides (golden and system annotations).

3) Coreference resolution (identifying mentions of the same entity)

Computed by the LEA metric: www.aclweb.org/anthology/P16-1060.pdf
Note: the importance of an entity is taken as log2 (number of mentions).

3a) at document level

3b) at single-language level

3c) at crosslingual level

Only cross-lingual links are considered by the metric. Entity weighting stays the same (based on the number of entity mentions). 
