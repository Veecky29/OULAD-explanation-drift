"""Phase 3 per-pair job: B=100 full-size bootstrap of BOTH the target (native models) and the source
(transfer model), so CIs include transfer-model uncertainty. One job = one (cutoff, model, kind, pair);
results are checkpointed to one JSON per job and skipped if present.
Usage: python src/phase3.py --cutoff 28 --model XGB --kind real --pair "LOMO EEE" [--B 100]
kind: real (LOMO / 2013->2014 per module, as phase 1/2) | null (within-module random halves, SELF-DESIGNED).
SELF-DESIGNED: the source+target bootstrap estimator (floor = sim(native_b, native_b+1); excess_b =
floor_b - sim(transfer_b, native_b)); 'fixed' variant keeps the point source model (phase-2 design).
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
sys.path.insert(0, os.path.dirname(__file__))
import experiment as E
from phase2 import Gran, zx, summarize, sim

GR = ["feature", "merged", "family"]; KS = ["spearman", "kendall", "topk", "sign"]


class Booter:
    def __init__(self, df, seed):
        self.g = df.groupby("id_student").indices; self.ids = np.array(list(self.g.keys())); self.rng = np.random.default_rng(seed)
    def draw(self):
        return np.concatenate([self.g[i] for i in self.rng.choice(self.ids, len(self.ids), replace=True)])


def make_pair(df, kind, pair):
    mo = pair.split()[-1] if pair.startswith(("LOMO", "NULL")) else pair.split()[0]
    if kind == "null":
        d = df[df.code_module == mo].reset_index(drop=True)
        i, j = next(GroupShuffleSplit(1, test_size=.5, random_state=E.SEED).split(d, d.y, d.id_student))
        return d.iloc[i].reset_index(drop=True), d.iloc[j].reset_index(drop=True)
    if pair.startswith("LOMO"):
        t = df[df.code_module == mo]; s = df[(df.code_module != mo) & (~df.id_student.isin(t.id_student))]
    else:
        s, t = df[(df.code_module == mo) & (df.year == 2013)], df[(df.code_module == mo) & (df.year == 2014)]
    return s.reset_index(drop=True), t.reset_index(drop=True)


def run(src, tgt, feats, kind_model, B, model_cls=E.Model, gran=None, seed=E.SEED):
    G = gran or Gran(feats); Zt = zx(tgt, feats)
    m = model_cls(kind_model, feats).fit(src, src.y)
    point = summarize(m.shap(tgt), Zt, G); auc_point = roc_auc_score(tgt.y, m.predict_proba(tgt))
    bs, bt = Booter(src, seed + 1), Booter(tgt, seed + 2)
    tr, nat, aucs = [], [], []
    for b in range(B):
        sb = src.iloc[bs.draw()]; mb = model_cls(kind_model, feats).fit(sb, sb.y)
        tr.append(summarize(mb.shap(tgt), Zt, G)); aucs.append(roc_auc_score(tgt.y, mb.predict_proba(tgt)))
        tb = tgt.iloc[bt.draw()]; nat.append(summarize(model_cls(kind_model, feats).fit(tb, tb.y).shap(tgt), Zt, G))
    out = dict(transfer_auc=auc_point, transfer_auc_lo=float(np.percentile(aucs, 2.5)), transfer_auc_hi=float(np.percentile(aucs, 97.5)),
               n_src=len(src), n_tgt=len(tgt), B=B)
    perb = {}
    for g in GR if gran is None else list(G.maps):
        for k in KS:
            fl = np.array([sim(nat[b][g], nat[(b + 1) % B][g], g)[k] for b in range(B)], float)
            tb_ = np.array([sim(tr[b][g], nat[b][g], g)[k] for b in range(B)], float)
            tf = np.array([sim(point[g], nat[b][g], g)[k] for b in range(B)], float)
            for var, t in [("boot", tb_), ("fixed", tf)]:
                ex = fl - t; f = np.isfinite(ex)
                out[f"{g}_{var}_excess_{k}"] = float(np.mean(ex[f])) if f.any() else None
                lo, hi = (np.percentile(ex[f], [2.5, 97.5]) if f.any() else (np.nan, np.nan))
                out[f"{g}_{var}_excess_{k}_lo"], out[f"{g}_{var}_excess_{k}_hi"] = float(lo), float(hi)
                out[f"{g}_{var}_sim_{k}"] = float(np.nanmean(t))
            out[f"{g}_floor_{k}"] = float(np.nanmean(fl))
            if k == "spearman":
                perb[g] = dict(floor=fl.tolist(), boot=tb_.tolist(), fixed=tf.tolist())
    out["perb"] = perb
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--kind", required=True); ap.add_argument("--pair", required=True); ap.add_argument("--B", type=int, default=100)
    a = ap.parse_args(); E.XGB_PARAMS["n_jobs"] = 2
    od = f"{E.ROOT}/results/phase3/b{a.B}/d{a.cutoff}"; os.makedirs(od, exist_ok=True)
    fn = f"{od}/{a.kind}_{a.model}_{a.pair.replace(' ', '_').replace('>', '').replace('-', '')}.json"
    if os.path.exists(fn): print("skip", fn); sys.exit(0)
    t0 = time.time(); df, feats = E.load(a.cutoff); s, t = make_pair(df, a.kind, a.pair)
    r = run(s, t, feats, a.model, a.B)
    r.update(cutoff=a.cutoff, model=a.model, kind=a.kind, pair=a.pair, family=("null" if a.kind == "null" else ("lomo" if a.pair.startswith("LOMO") else "cross_year_module")),
             seconds=time.time() - t0)
    json.dump(r, open(fn + ".tmp", "w")); os.replace(fn + ".tmp", fn)
    print(f"done {fn} {time.time()-t0:.0f}s feat_boot {r['feature_boot_excess_spearman']:.3f} fam_boot {r['family_boot_excess_spearman']:.3f}", flush=True)
