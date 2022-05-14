from math import floor
from random import uniform
from typing import List

from keras.models import Sequential, load_model
from keras.layers import Activation, Dense
from keras.utils.np_utils import to_categorical

import numpy as np

import utils.plots as plots


def to_index(onehot: list) -> str:
    max_val = 0
    max_index = 0
    for i in range(0, len(onehot)):
        if onehot[i] > max_val:
            max_val = onehot[i]
            max_index = i
    return max_index


def generate_data(n_ins, n_outs, dict, presc: int = 5000) -> tuple((List, List)):
    print(">> Preparing model")
    inputs = [round(uniform(0, n_ins), 2) for i in range(0, presc)]
    targets = [to_categorical(dict[floor(i)], n_outs) for i in inputs]
    return np.asarray(inputs), np.asarray(targets)


def build_model(n_ins, input_shape, n_outs) -> Sequential:
    print(">> Building model")
    model = Sequential(
        [
            Dense(n_ins, input_shape=input_shape, activation="swish"),
            Dense(n_outs),
            Activation("softmax"),
        ]
    )
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    return model


def train(x, y, model, batch_size=128, epochs=50, verbose=1, model_path="./model.h5", plot_name="model"):
    print(">> Training model")
    history = model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=verbose)
    model.save(model_path)
    print(history.history.keys())
    plots.acc(history.history, plot_name)
    plots.loss(history.history, plot_name)
    return model


def load(model_path):
    print(">> Loading model")
    return load_model(model_path)
