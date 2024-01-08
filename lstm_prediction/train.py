import io
import json
import re
import subprocess

import dvc.api
import fire
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from hydra import compose
from hydra.experimental import initialize
from tensorflow.keras.layers import Embedding


def prepare(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text


def prepare_data():
    repo = "https://github.com/NGeraskina/ML_Ops_part2"
    with dvc.api.open("data/medium_data.csv", repo=repo, encoding="utf-8") as file:
        df = pd.read_csv(file, parse_dates=["date"])
    df.title = df.title.apply(lambda x: prepare(x))
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<oov>")
    tokenizer.fit_on_texts(df.title)
    total_words = len(tokenizer.word_index) + 1
    input_sequences = []
    for line in df.title:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[: i + 1]
            input_sequences.append(n_gram_sequence)

    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(
        tf.keras.preprocessing.sequence.pad_sequences(
            input_sequences, maxlen=max_sequence_len, padding="pre"
        )
    )

    X, y = input_sequences[:, :-1], input_sequences[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)

    tokenizer_json = tokenizer.to_json()
    with io.open("models/tokenizer.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    return X, y, total_words, max_sequence_len


def train_model(cfg) -> None:
    with mlflow.start_run():
        mlflow.set_tracking_uri(f"http://{cfg.train.host}:{cfg.train.port}")

        git_commit_id = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .strip()
            .decode("utf-8")
        )

        mlflow.log_param("git_commit_id", git_commit_id)
        mlflow.log_param("hidden_layer", cfg.train.hidden_layer)

        X, y, total_words, max_sequence_len = prepare_data()
        # def train(X, y, total_words, hidden_layer = 128, activation = 'softmax', lr = 0.001):
        # X, y = pd.read_csv('X.csv'), pd.read_csv('y.csv')
        hidden_layer, activation, lr, epoch = (
            cfg.train.hidden_layer,
            cfg.train.activation,
            cfg.train.lr,
            cfg.train.epoch,
        )

        model = tf.keras.models.Sequential()
        model.add(
            Embedding(total_words, hidden_layer, input_length=max_sequence_len - 1)
        )
        model.add(tf.keras.layers.LSTM(hidden_layer))
        model.add(tf.keras.layers.Dense(total_words, activation=activation))
        model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            metrics=["accuracy", "mse"],
        )

        history = model.fit(X, y, epochs=epoch)

        for i in range(epoch):
            mlflow.log_metric("Accuracy_train", history.history["accuracy"][i], step=i)
            mlflow.log_metric("MSE_train", history.history["mse"][i], step=i)
            mlflow.log_metric(
                "Categorical_crossentropy_train", history.history["loss"][i], step=i
            )

        model.save(cfg.train.model_output_path)


if __name__ == "__main__":
    with initialize(config_path="../configs"):
        cfg = compose(config_name="model_config")

    fire.Fire(train_model(cfg))
