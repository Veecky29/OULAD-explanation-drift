"""Compute v1/v2 trust flags for every OULAD real pair and null pair at all 5 cutoffs (point source model)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import experiment as E
from phase3 import make_pair
from trust import trust_metrics

E.XGB_PARAMS["n_jobs"] = 2
out = f"{E.ROOT}/results/phase3/trust_oulad.csv"; rows = []
for c in [7, 14, 28, 56, 112]:
    df, feats = E.load(c)
    for m in ["LR", "XGB"]:
        for kind, pairs in [("real", [f"LOMO {x}" for x in "AAA BBB CCC DDD EEE FFF GGG".split()] + [f"{x} 13->14" for x in "AAA BBB DDD EEE FFF GGG".split()]),
                            ("null", [f"NULL {x}" for x in "AAA BBB CCC DDD EEE FFF GGG".split()])]:
            for p in pairs:
                s, t = make_pair(df, kind, p)
                mod = E.Model(m, feats).fit(s, s.y); ss = s.sample(min(len(s), 6000), random_state=E.SEED)
                imp = mod.shap(ss).abs().mean().drop("region", errors="ignore")
                r = trust_metrics(s, t, list(imp.nlargest(5).index)); r.update(cutoff=c, model=m, kind=kind, pair=p); rows.append(r)
        pd.DataFrame(rows).to_csv(out, index=False); print("done", c, m, flush=True)
