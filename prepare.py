import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("./jasp/sp.model")


def execute(testdata, outprefix):
    df = pd.read_csv(testdata, sep="\t")
    df_p = df[df["label"] == 1]
    df_n = df[df["label"] == 0]

    qvecs = tf.keras.preprocessing.sequence.pad_sequences(
        [sp.EncodeAsIds(str(x)) for x in df_p["sentence1"]], maxlen=300)
    qvecs_n = tf.keras.preprocessing.sequence.pad_sequences(
        [sp.EncodeAsIds(str(x)) for x in df_n["sentence1"]], maxlen=300)
    
    np.save("qvecs_{}.npy".format(outprefix), qvecs)
    np.save("qvecs_n_{}.npy".format(outprefix), qvecs_n)
    
    dvecs = tf.keras.preprocessing.sequence.pad_sequences(
        [sp.EncodeAsIds(str(x)) for x in df_p["sentence2"]], maxlen=300)
    dvecs_n = tf.keras.preprocessing.sequence.pad_sequences(
        [sp.EncodeAsIds(str(x)) for x in df_n["sentence2"]], maxlen=300)

    np.save("dvecs_{}.npy".format(outprefix), dvecs)
    np.save("dvecs_n_{}.npy".format(outprefix), dvecs_n)


if __name__ == "__main__":
    execute("../data/pawsx/dev.tsv", "dev_features")
    execute("../data/pawsx/train.tsv", "train_features")
    execute("../data/pawsx/test.tsv", "test_features")
