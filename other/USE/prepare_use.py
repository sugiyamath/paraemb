import tensorflow_hub as hub
import tensorflow_text
import numpy as np
import pandas as pd


def load_data(infile):
    df = pd.read_csv(infile, sep="\t")
    s1s = list(map(str, df["sentence1"]))
    s2s = list(map(str, df["sentence2"]))
    y = df["label"]
    return df, s1s, s2s, y


def encode(ss):
    embed = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
    return embed(ss)


def cossim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def simall(v1s, v2s):
    return [cossim(v1, v2) for v1, v2 in zip(v1s, v2s)]


def main():
    infiles = ["../data/pawsx/train.tsv", "../data/pawsx/test.tsv"]

    outfiles = ["./train_fx.tsv", "./test_fx.tsv"]
    for outfile, infile in zip(outfiles, infiles):
        df, s1s, s2s, y = load_data(infile)
        v1s, v2s = encode(s1s), encode(s2s)
        df["usesim"] = simall(v1s, v2s)
        df.to_csv(outfile, index=False, sep="\t")


if __name__ == "__main__":
    main()
