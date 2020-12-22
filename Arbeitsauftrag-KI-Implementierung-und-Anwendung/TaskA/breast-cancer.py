import numpy as np
import pandas as pd
from KNN import KNN


def main():
    df = pd.read_csv("./data/cancer.csv")
    df.drop(['id'], axis=1)  # ignore id, not relevant for diagnosis

    def diagnosis_value(diagnosis):  # Convert Diagnosis value (Malignant (M) =1) and Benign (B) =0
        return 1 if diagnosis == "M" else 0

    df["diagnosis"] = df["diagnosis"].apply(diagnosis_value)

    X = np.array(df.iloc[:, 1:])  # select breast cancer indicators (radius, texture, perimeter, smoothness)
    Y = np.array(df["diagnosis"])  # diagnosis

    knn = KNN()
    percent=int(0.8*len(X))
    knn.fit(X[:percent], Y[:percent])  # use 80% of data for training

    preds = knn.predict(X[percent:])
    expected = Y[percent:]
    for i, pred in enumerate(preds):
        print("Predicted " + str(pred) + ", Expected " + str(expected[i]))


if __name__ == '__main__':
    main()
