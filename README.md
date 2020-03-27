# paraemb
An experiment of creating sentence embedding from paraphrase corpus

Currently, this project is just an experiment.

## what's this?

This project is an experiment of sentence embedding. The motivation is solving paraphrase identification task only by sentence vector's similarity.

## requirements

- sentencepiece
- tensorflow-gpu >= 2.0.0
- numpy
- pandas


## execution flow

prepare.py -> models/<modelname> -> run some scripts

### prepare.py

prepare.py is a data preparation script. To use this script, you need to prepare an paraphrase corpus, like PAWS-X.

The data format is this:

||col1|col2|col3|
|---|---|---|---|
|colname|sentence1|sentence2|label|
|description|the input sentence of the model|pair sentence of input|0: hard-negative, 1:hard-positive|

execute function in prepare.py has these params:

```
execute(datapath, output_prefix)
```

At least, you need an validation set and an training set.

The outputs are this:

- qvecs_features.npy: Input vectors for hard-potitives.
- qvecs_n_features.npy: Input vectors for hard-negatives.
- dvecs_features.npy: hard-positive vectors that were transformed from sentence2.
- dvecs_n_features.npy: hard-negative vectors that were transformed from sentence2.


## TODO

- test sentence-bert : https://github.com/UKPLab/sentence-transformers