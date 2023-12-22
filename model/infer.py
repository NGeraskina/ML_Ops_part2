import json

import numpy as np
import tensorflow as tf
import pandas as pd
import fire

from hydra import compose
from hydra.experimental import initialize

from keras.models import load_model


def predict(cfg) -> None:
    next_words, model_path, max_sequence_len = (
        cfg.infer.next_words,
        cfg.infer.model_path,
        cfg.infer.max_sequence_len,
    )

    X = pd.read_csv('test_data.csv')

    model = load_model('trained_model.h5')

    pred = []

    for text in X['title_new']:
        for _ in range(next_words):
            with open("tokenizer.json") as f:
                data = json.load(f)
                tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)
            token_list = tokenizer.texts_to_sequences([text])[0]
            token_list = tf.keras.preprocessing.sequence.pad_sequences(
                [token_list], maxlen=max_sequence_len - 1, padding="pre"
            )
            predicted = np.argmax(model.predict(token_list), axis=1)
            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    text += word + " "
        pred.append(text)

    X['predict'] = pred
    X.to_csv('test_data_infered.csv')
    return None


if __name__ == "__main__":
    with initialize(config_path="../config"):
        cfg = compose(config_name="model_config")

    fire.Fire(predict(cfg))