"""Null control for the bootstrap noise floor (SELF-DESIGNED).
Within each module, split students 50/50 (student-grouped, seed 42): source = half A, target = half B.
There is no real course shift, so any 'excess drift' here measures the bias of the bootstrap floor
(plus the halved source size). Usage: python src/phase2_null.py --cutoff 28
"""
import argparse, os, sys, time
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
sys.path.insert(0, os.path.dirname(__file__))
import experiment as E, phase2 as P

ap = argparse.ArgumentParser(); ap.add_argument("--cutoff", type=int, default=28); ap.add_argument("--B", type=int, default=30)
a = ap.parse_args(); E.XGB_PARAMS["n_jobs"] = 2
df, feats = E.load(a.cutoff); out = f"{E.ROOT}/results/phase2/null_d{a.cutoff}"; os.makedirs(out, exist_ok=True)
rows = []
for kind in ["LR", "XGB"]:
    for mo in sorted(df.code_module.unique()):
        d = df[df.code_module == mo].reset_index(drop=True)
        i, j = next(GroupShuffleSplit(1, test_size=.5, random_state=E.SEED).split(d, d.y, d.id_student))
        t0 = time.time()
        r, _, _ = P.run_pair(f"NULL {mo}", "null_within_module", d.iloc[i].reset_index(drop=True), d.iloc[j].reset_index(drop=True), feats, kind, a.B, a.cutoff)
        rows.append(r); pd.DataFrame(rows).to_csv(f"{out}/pairs.csv", index=False)
        print(f"[null d{a.cutoff}] {kind} {mo} feat excess {r['feature_excess_spearman']:.2f} fam {r['family_excess_spearman']:.2f} ({time.time()-t0:.0f}s)", flush=True)
print("DONE", flush=True)
