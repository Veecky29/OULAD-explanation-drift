"""Explanation noise floor: how similar are SHAP rankings of two models trained on the SAME domain?
For each target used in H2 (LOMO modules, per-module 2014 targets, pooled 2014), split target students
into two halves (student-grouped, 5 repetitions), train a native model on each half, and compute the
Spearman / Jaccard between their mean|SHAP| on the full target. If transfer-vs-native similarity is
clearly below this split-half reliability, the drift exceeds what sampling noise alone produces.
Caveat: each half has ~50% of the native training data, so the floor is conservative (noisier).
Usage: python src/noise_floor.py --cutoff 28
"""
import argparse, os, sys
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
sys.path.insert(0, os.path.dirname(__file__))
from experiment import load, Model, expl_summary, compare, ROOT


def main(c, reps=5):
    df, feats = load(c)
    targets = [("Y13->Y14 pooled", df[df.year == 2014])]
    for mo in sorted(df.code_module.unique()):
        t = df[(df.code_module == mo) & (df.year == 2014)]
        if len(df[(df.code_module == mo) & (df.year == 2013)]): targets.append((f"{mo} 13->14", t))
    for mo in sorted(df.code_module.unique()):
        targets.append((f"LOMO {mo}", df[df.code_module == mo]))
    rows = []
    for kind in ["LR", "XGB"]:
        for name, t in targets:
            t = t.reset_index(drop=True)
            for r in range(reps):
                a, b = next(GroupShuffleSplit(1, test_size=0.5, random_state=r).split(t, t.y, t.id_student))
                ma = Model(kind, feats).fit(t.iloc[a], t.y.iloc[a]); mb = Model(kind, feats).fit(t.iloc[b], t.y.iloc[b])
                iA, dA = expl_summary(ma.shap(t), t, feats); iB, dB = expl_summary(mb.shap(t), t, feats)
                d = compare(iA, dA, iB, dB); d.update(model=kind, pair=name, rep=r, n_tgt=len(t)); rows.append(d)
            print(kind, name, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(f"{ROOT}/results/d{c}/noise_floor_split_half.csv", index=False)
    agg = out.groupby(["model", "pair"])[["spearman", "kendall", "jac5", "jac10", "sign_agree"]].mean().reset_index()
    pr = pd.read_csv(f"{ROOT}/results/d{c}/transfer_pairs.csv")
    m = agg.merge(pr[["model", "pair", "family", "tgtVSnative_spearman", "tgtVSnative_jac10", "tgtVSnative_sign_agree",
                      "auc_drop"]], on=["model", "pair"])
    m["excess_drift"] = m["spearman"] - m["tgtVSnative_spearman"]
    m.to_csv(f"{ROOT}/results/d{c}/noise_floor_vs_transfer.csv", index=False)
    print(m.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cutoff", type=int, default=28); main(ap.parse_args().cutoff)
