import sys
import numpy as np
import tensorflow as tf
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("./jasp/sp.model")

model = tf.keras.models.load_model("./model_de2.h5")
model_fx = tf.keras.models.Model(inputs=model.get_layer("in1").input,
                                 outputs=model.get_layer("sentemb").output)


def encode(texts):
    X = tf.keras.preprocessing.sequence.pad_sequences(
            [sp.EncodeAsIds(str(x)) for x in texts], maxlen=300)
    return model_fx.predict(X)


def cossim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


if __name__ == "__main__":
    texts = ["これはテスト",
             "テストです",
             "物理学は楽しい",
             "数学は楽しい",
             "この本は興味深い",
             "興味深い本だ"]
    vs = encode(texts)
    for i, v1 in enumerate(vs):
        for j, v2 in enumerate(vs):
            print(texts[i], texts[j], cossim(v1, v2))
