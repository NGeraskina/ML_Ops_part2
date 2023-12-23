# Generating the title of an article based on the first few words

Project goal: train an LSTM neural network using headlines from Medium.com and generate an article title based on the
first words

## Initial configuration

`setup.py` is planned, for now manual installation

For usage:

1. install `poetry`
2. run `poetry lock`

For development:

- install `dvc`
- install `pre-commit`
- run `pre-commint install`
- run `dvc pull` to get all datasets and models

# General scheme

1. Preprocessing: make lower case, delete unwanted spaces, train tokenizer and tokenize input data
2. LSTM training: NN consists of Embedding, LSTM and Dense layers 