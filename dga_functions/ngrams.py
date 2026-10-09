import argparse
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from features import FEATURE_NAMES

def true_pos_at_false_pos_rate(y_true, scores, target_fpr):
    fpr, tpr, thr = roc_curve(y_true, scores)
    i = np.searchsorted(fpr ,target_fpr, side="right") - 1
    return tpr[max(i, 0)], thr[max(i, 0)]

def make_model():
    features = ColumnTransformer([
        ("ngrams", TfidfVectorizer(analyzer="char", ngram_range=(2,4), min_df=2, sublinear_tf=True), "label"),
        ("handmade", StandardScaler(), FEATURE_NAMES),
    ])
    return Pipeline([
        ("features", features),
        ("model", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))
    ])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataset.csv")
    p.add_argument("--fpr", type=float, default=0.001)
    a = p.parse_args()
    df = pd.read_csv(a.data, keep_default_na=False)
    benign = df[df.is_dga == 0].sample(frac=1, random_state=0)
    cut = int(len(benign) * .8)
    benign_train, benign_test = benign.iloc[:cut], benign.iloc[cut:] 

    print(f"{'held-out family':16s} {'AUC':>7s} {'TPR@1%FPR':>10s} {'TPR@' + str(a.fpr * 100) + '%FPR':>12s}")
    for fam in sorted(df.loc[df.is_dga == 1, "family"].unique()):
        dga = df[df.is_dga == 1]
        train = pd.concat([dga[dga.family != fam], benign_train])
        test = pd.concat([dga[dga.family == fam], benign_test])
        model = make_model()
        model.fit(train, train.is_dga)
        scores = model.predict_proba(test)[:, 1]
        auc = roc_auc_score(test.is_dga, scores)
        tpr1, _ = true_pos_at_false_pos_rate(test.is_dga, scores, 0.01)
        tprStrict, _ = true_pos_at_false_pos_rate(test.is_dga, scores, a.fpr)
        print(f"{fam:16s} {auc:7.3f} {tpr1:10.1%} {tprStrict:12.1%}")

if __name__== "__main__":
    main()