import argparse
import pandas as pd
from features import label_of, handmade

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dga", required=True)
    p.add_argument("--benign", required=True)
    p.add_argument("--out", default="dataset.csv")
    p.add_argument("--benign-taker", type=int, default=50000, help="benign domains from the list")
    a = p.parse_args()

    dga = pd.read_csv(a.dga)
    dga["is_dga"] = 1

    benign = pd.read_csv(a.benign, header=None, names=["rank", "domain"]).head(a.benign_taker)
    benign = benign[["domain"]].assign(family="benign", is_dga=0)
    df = pd.concat([dga, benign], ignore_index=True)
    df["label"] = df["domain"].map(label_of)
    df = df[df["label"].str.len() > 0].drop_duplicates(subset=["label", "is_dga"])

    clash = set(df.loc[df.is_dga == 1, "label"]) & set(df.loc[df.is_dga == 0, "label"])
    if clash:
        print(f"dropping {len(clash)} appearing in both classes")
        df = df[~df["label"].isin(clash)]

    feats = pd.DataFrame([handmade(label) for label in df["label"]], index=df.index)
    df = pd.concat([df, feats], axis=1)
    df.to_csv(a.out, index=False)
    print(df["family"].value_counts().to_string())
    print(f"\nwrote {len(df)} rows --- {a.out}")

if __name__ == "__main__":
    main()