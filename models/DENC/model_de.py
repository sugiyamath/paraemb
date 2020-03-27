import os
import sys
import random

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.utils import shuffle
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, SeparableConv1D, GlobalMaxPooling1D, MaxPooling1D, Input, Dense, Dropout, Flatten, BatchNormalization, Multiply, LSTM

bs = 10
lr = 1e-3
margin = 0.5
epoch = 2
logs = "./logs_de/"


def build_data(dname, prefix):
    ft1 = os.path.join(dname, "qvecs_{}_features.npy".format(prefix))
    ft2 = os.path.join(dname, "dvecs_{}_features.npy".format(prefix))
    fn1 = os.path.join(dname, "qvecs_n_{}_features.npy".format(prefix))
    fn2 = os.path.join(dname, "dvecs_n_{}_features.npy".format(prefix))
    X1_t = np.load(ft1)
    X2_t = np.load(ft2)
    X1_n = np.load(fn1)
    X2_n = np.load(fn2)
    X1_f = shuffle(X1_t)
    X2_f = shuffle(X2_t)
    X1 = np.vstack([X1_t, X2_t, X1_n, X2_n, X1_f])
    X2 = np.vstack([X2_t, X1_t, X2_n, X1_n, X2_f])
    #X1 = np.vstack([X1_t, X2_t, X1_n, X2_n])
    #X2 = np.vstack([X2_t, X1_t, X2_n, X1_n])
    y = []
    y += [1 for _ in range(X1_t.shape[0] + X2_t.shape[0])]
    y += [0 for _ in range(X1_n.shape[0] + X2_n.shape[0])]
    y += [0 for _ in range(X1_f.shape[0])]
    y = np.array(y)
    X1, X2, y = shuffle(X1, X2, y)
    return [X1, X2], y


def _build(x, arr):
    out = x
    for a in arr:
        out = a(out)
    return out


def build_model(
    max_features=8000,
    max_len=300,
    dim1=50,
    dim2=200,
    drate=0.3,
    depth=2,
    learning_rate=lr,
):

    in1 = Input(shape=(300, ), name="in1")
    in2 = Input(shape=(300, ), name="in2")
    emb = Embedding(max_features + 1,
                    dim1,
                    input_length=max_len,
                    embeddings_initializer="he_normal",
                    mask_zero=True,
                    trainable=True,
                    name="emb")
    sentemb = LSTM(units=dim2, name="sentemb")
    x1 = sentemb(emb(in1))
    x2 = sentemb(emb(in2))
    out = Multiply()([x1, x2])
    out = Dense(1, activation="sigmoid")(out)
    model = Model([in1, in2], out)
    model.compile(loss="binary_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(lr=learning_rate))
    return model


if __name__ == "__main__":
    X_train, y_train = build_data("./prepared_data", "train")
    X_dev, y_dev = build_data("./prepared_data", "dev")
    model = build_model()
    model.fit(
        X_train,
        y_train,
        batch_size=bs,
        epochs=epoch,
        validation_data=(X_train, y_train),
        shuffle=True,
        callbacks=[tf.keras.callbacks.TensorBoard(logs, update_freq="batch")])
    model.save("model_de.h5")
