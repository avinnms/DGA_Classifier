import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from features import FEATURE_NAMES

def true_pos_at_false_pos_rate(y_true, scores, target_fpr):
    fpr, tpr, thr = roc_curve(y_true, scores)
    i = np.searchsorted(fpr ,target_fpr, side="right") - 1
    return tpr[max(i, 0)], thr[max(i, 0)]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dga_domains.csv")
    p.add_argument("--fpr", type=float, default=0.001, help="shows false positive rate, where .001 is .1%% false positive")
    a = p.parse_args()
    df = pd.read_csv(a.data)
    benign = df[df.is_dga == 0].sample(frac=1, random_state=0)
    cut = int(len(benign) * .8)
    benign_train, benign_test = benign.iloc[:cut], benign.iloc[cut:]

    print(f"{'held-out family':16s} {'AUC':>7s} {'TPR@1%FPR':>10s} {'TPR@' + str(a.fpr * 100) + '%FPR':>12s}")
    for fam in sorted(df.loc[df.is_dga == 1, "family"].unique()):
        dga = df[df.is_dga == 1]
        train = pd.concat([dga[dga.family != fam], benign_train])
        test = pd.concat([dga[dga.family == fam], benign_test])
        model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0, class_weight="balanced")
        model.fit(train[FEATURE_NAMES], train.is_dga)
        scores = model.predict_proba(test[FEATURE_NAMES])[:, 1]

        auc = roc_auc_score(test.is_dga, scores)
        tpr1, _ = true_pos_at_false_pos_rate(test.is_dga, scores, 0.01)
        tpr_strict, _ = true_pos_at_false_pos_rate(test.is_dga, scores, a.fpr)
        print(f"{fam:16s} {auc:7.3f} {tpr1:10.1%} {tpr_strict:12.1%}")

if __name__ == "__main__":
    main()
