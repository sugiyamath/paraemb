import sys
import random

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.utils import shuffle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, SeparableConv1D, GlobalMaxPooling1D, MaxPooling1D, Input, Dense, Dropout, Flatten, BatchNormalization, GRU

bs = 5
lr = 1e-6
margin = 0.5
epoch = 1
logs = "./logs_hn/"


X_t = np.load("./prepared_data/qvecs_train_features.npy")
Y_t = np.load("./prepared_data/dvecs_train_features.npy")
X_n = np.load("./prepared_data/qvecs_n_train_features.npy")
Y_n = np.load("./prepared_data/dvecs_n_train_features.npy")

X_train = np.vstack([X_t, Y_t])
Y_train = np.vstack([Y_t, X_t])

tsize = (X_train.shape[0] // bs) * bs
X_train = X_train[:tsize]
Y_train = Y_train[:tsize]
Y_rand = shuffle(Y_train)
print(X_train.shape)
print(Y_train.shape)
X_d = np.load("./prepared_data/qvecs_dev_features.npy")
Y_d = np.load("./prepared_data/dvecs_dev_features.npy")
X_dev = np.vstack([X_d, Y_d])
Y_dev = np.vstack([Y_d, X_d])


def generate_vdata():
    while True:
        x, y = shuffle(X_dev, Y_dev)
        yield x[:bs], y[:bs]


def dummy_metrics(x, y):
    return 0


def triplet_loss(model, X_n, Y_n, margin):
    def pred(model, x):
        x_cp = x
        for i in range(len(model.layers)):
            x_cp = model.layers[i](x_cp)
        return x_cp

    def loss(y_true, y_pred):
        ns_x, ns_y = shuffle(X_n, Y_n)
        ns_x = ns_x[:bs]
        ns_y = ns_y[:bs]
        va = pred(model, y_true)
        vp = y_pred
        vn_x = pred(model, np.array(ns_x))
        vn_y = pred(model, np.array(ns_y))
        dx = tf.norm(va - vp)
        dy = tf.norm(vn_x - vn_y)
        T = dx - dy
        T = tf.add(T, tf.constant(margin))
        T = tf.maximum(T, 0.0)
        valid_triplets = tf.cast(tf.math.greater(T, 1e-16), dtype=tf.float32)
        num_positive_triplets = tf.reduce_sum(valid_triplets)
        T = tf.reduce_sum(T) / (num_positive_triplets + 1e-16)
        return T

    return loss


def trainB(model, X, y, loss_function, epochs=3, outdir="./logs"):
    writer = tf.summary.create_file_writer(logdir=outdir)

    with writer.as_default():
        for j in range(epochs):
            print("epoch:", j + 1)
            for i in range(X.shape[0] // bs):
                X_t = X[i * bs:(i + 1) * bs]
                y_t = y[i * bs:(i + 1) * bs]
                loss = model.train_on_batch(X_t, y_t)[0]
                tf.summary.scalar("train_loss", loss,
                                  i + (X.shape[0] // bs) * j)
                writer.flush()
                if i % 10 == 0:
                    sys.stdout.write("Iter:{0: >5}, loss:{1: >20}\r".format(
                        i, loss))
                    sys.stdout.flush()
            print("\nloss:{}".format(
                model.test_on_batch(*next(generate_vdata()))))


def build_model(data,
                margin=0.5,
                max_features=8000,
                max_len=300,
                dim1=50,
                dim2=200,
                drate=0.25,
                learning_rate=lr):
    model = Sequential()
    model.add(Input(shape=(300, )))
    model.add(
        Embedding(max_features + 1,
                  dim1,
                  input_length=max_len,
                  embeddings_initializer="he_normal"))
    model.add(Flatten())
    model.add(Dense(dim2))
    model.add(Dropout(drate))
    model.add(Dense(dim2))
    model.add(Dropout(drate))
    model.add(Dense(dim2))
    lf = triplet_loss(model, X_n, Y_n, margin)
    model.compile(loss=lf,
                  optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  metrics=[dummy_metrics])
    return model, lf


if __name__ == "__main__":
    model, lf = build_model(X_train, margin=margin)
    trainB(model, X_train, Y_train, lf, epochs=epoch, outdir=logs)
    model.save("model_hn.h5")
