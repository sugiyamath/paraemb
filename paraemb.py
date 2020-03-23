import sys
import numpy as np
import tensorflow as tf
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("./jasp/sp.model")

model = tf.keras.models.load_model(sys.argv[1], compile=False)


def encode(texts):
    return model.predict(
        tf.keras.preprocessing.sequence.pad_sequences(
            [sp.EncodeAsIds(str(x)) for x in texts], maxlen=300))


def cossim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


if __name__ == "__main__":
    texts = ["これはテスト", "テストです", "物理学は楽しい", "数学は楽しい"]
    vs = encode(texts)
    #print(vs)
    for i, v1 in enumerate(vs):
        for j, v2 in enumerate(vs):
            print(texts[i], texts[j], cossim(v1, v2))
