import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def cossim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def _calc_features(df, enc1, enc2):
    oL = []
    oP = []
    v1s_L = enc1('\n'.join([str(x).replace("\n", "") for x in df["sentence1"]]))
    v2s_L = enc1('\n'.join([str(x).replace("\n", "") for x in df["sentence2"]]))
    v1s_P = enc2(df["sentence1"])
    v2s_P = enc2(df["sentence2"])
    assert v1s_L.shape[0] == df.shape[0]
    assert v2s_L.shape[0] == df.shape[0]
    assert v1s_P.shape[0] == df.shape[0]
    assert v2s_P.shape[0] == df.shape[0]
    for v1_P, v2_P, v1_L, v2_L in tqdm(zip(v1s_P, v2s_P, v1s_L, v2s_L)):
        oL.append(cossim(v1_L, v2_L))
        oP.append(cossim(v1_P, v2_P))
    df["laser"] = oL
    df["paraemb"] = oP
    return df


def main(infile, outfile):
    import paraemb
    import laserencoder
    enc = laserencoder.Encoder()
    df = pd.read_csv(infile, sep="\t")
    df = _calc_features(df, enc.encode, paraemb.encode)
    df.to_csv(outfile, sep="\t", index=False)


def test(trainfile, testfile):
    cnames = [["laser"], ["paraemb"], ["laser", "paraemb"]]

    for cname in cnames:
        df = pd.read_csv(trainfile, sep="\t")
        X_train = df[cname]
        y_train = df["label"]
        df = pd.read_csv(testfile, sep="\t")
        X_test = df[cname]
        y_test = df["label"]
        clf = DecisionTreeClassifier(max_leaf_nodes=8)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("[{}]".format(cname))
        print(classification_report(y_test, y_pred))
        print()


if __name__ == "__main__":
    #main("./test.tsv", "./test_result.tsv")
    #main("../data/pawsx/train.tsv", "./train_result.tsv")
    test("./train_result.tsv", "./test_result.tsv")
