import datetime as dt
import json

import dvc.api
import fire
import numpy as np
import pandas as pd
import tensorflow as tf
from hydra import compose
from hydra.experimental import initialize
from keras.models import load_model


def predict(cfg) -> None:
    next_words, model_path, max_sequence_len = (
        cfg.infer.next_words,
        cfg.infer.model_path,
        cfg.infer.max_sequence_len,
    )

    repo = "https://github.com/NGeraskina/ML_Ops_part2"

    with dvc.api.open("data/test_data.csv", repo=repo, encoding="utf-8") as test:
        X = pd.read_csv(test)

    dvc.api.DVCFileSystem(repo).get(model_path, lpath="model.h5")
    model = load_model("model.h5")

    pred = []
    for text in X["title_new"]:
        text += " "
        for _ in range(next_words):
            with dvc.api.open("../models/tokenizer.json", repo=repo) as f:
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

    X["predict"] = pred
    X.to_csv(
        f"predicted/test_data_infered_{dt.datetime.now().strftime('%Y_%m_%d_%H_%M')}.csv"
    )
    return None


if __name__ == "__main__":
    with initialize(config_path="../configs"):
        cfg = compose(config_name="model_config")
    fire.Fire(predict(cfg))
