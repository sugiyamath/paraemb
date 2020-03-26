import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report


def train(X, y):
    model = DecisionTreeClassifier(max_leaf_nodes=2)
    model.fit(X, y)
    export_graphviz(model, "tree.dot")
    return model


def test(model, X, y):
    y_pred = model.predict(X)
    print(classification_report(y, y_pred))


def main():
    df_train = pd.read_csv("./train_fx.tsv", sep="\t")
    df_test = pd.read_csv("./test_fx.tsv", sep="\t")
    X_train = df_train[["usesim"]]
    y_train = df_train["label"]
    X_test = df_test[["usesim"]]
    y_test = df_test["label"]

    model = train(X_train, y_train)
    test(model, X_test, y_test)


if __name__ == "__main__":
    main()
