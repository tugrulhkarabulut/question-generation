# Automatic Question Generation with Encoder-Decoder LSTM

This is an implementation of Encoder-Decoder LSTM with Bahdanau Attention to
automatic question generation task. SQuAD is used in training and testing.
If you want to learn about the details and results, check out my 
[project thesis](https://github.com/tugrulhkarabulut/question-generation/blob/master/project-thesis.pdf).


## How to run

First, download SQuAd from [here](https://rajpurkar.github.io/SQuAD-explorer/). Then, download GloVe from here [here](https://nlp.stanford.edu/projects/glove/)

Code consists of several modules. You can type the
folowing command to learn about the parameters of the scripts:
```bash
python3 modules/{filename}.py --help
```

You have these modules in the following order:

Preprocessing the dataset:
```bash
python3 modules/squad.py --input path-to-squad-json-file --out desired-output-path --out_format pkl_or_csv
```

Building the tokenizers:

```bash
python3 modules/tokenizer.py --input path-to-preprocessed-squad-file
```

You can customize the tokenizing process. Refer to the
help command to know about the parameters you can play with.

Training:

```bash
python3 modules/train.py --train_config path-to-train-config-file
```
You can define the training parameters (batch size, learning rate etc.) by modifying the
train_config.json file.

Inference:

```bash
python3 modules/inference.py --input path-to-tokenized-test-data
```

You can decide the decoding method and other things by
defining the parameters as well.
