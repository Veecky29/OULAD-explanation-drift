"""Phase 2: feature-family SHAP, full-sample bootstrap noise floor, more cutoffs, distribution shift,
pre-deployment 'explanation trust' check, calibration.
Usage: python src/phase2.py --cutoff 28 [--variant base|novisits] [--B 30] [--jobs 3]

Literature-aligned settings are inherited from experiment.py (see results.md section 2).
SELF-DESIGNED (labelled as such in results.md): sub-families of activity types, the bootstrap
noise-floor estimator, shift-metric aggregation, the trust-check rule, ECE binning (10 equal-width).
"""
import argparse, os, sys, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau, rankdata, ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
sys.path.insert(0, os.path.dirname(__file__))
import experiment as E
from experiment import Model, load, ROOT, SEED

# Top-level families follow Lim (2026) feature specifications (registration / VLE / submission / score /
# demographic) and Wu et al. (2026) behavior vs background. Sub-grouping of the 20 VLE activity types into
# content / quiz-like / communication / navigation is SELF-DESIGNED.
ACT_GROUP = {
    "content": ["oucontent", "resource", "url", "page", "subpage", "glossary", "dualpane", "htmlactivity",
                "folder", "dataplus", "repeatactivity"],
    "quiz_like": ["quiz", "externalquiz", "questionnaire"],
    "communication": ["forumng", "oucollaborate", "ouelluminate", "ouwiki"],
    "navigation": ["homepage", "sharedsubpage"],
}
COUNT_PREFIX = ("clicks_", "visits_", "total_", "precourse", "active_days", "n_submitted", "n_late", "days_since")


def family_of(f):
    if f in ("gender", "region", "highest_education", "imd_band", "age_band", "disability"): return "demographic"
    if f in ("date_registration", "num_of_prev_attempts", "studied_credits"): return "registration"
    if f in ("n_submitted", "n_late"): return "submission"
    if f == "mean_score": return "score"
    if f.startswith(("clicks_", "visits_")) and not f.startswith("clicks_week"):
        t = f.split("_", 1)[1]
        for g, ts in ACT_GROUP.items():
            if t in ts: return "vle_" + g
    return "vle_overall"


def merged_of(f):
    """Collinear clicks_X / visits_X merged into act_X; total_clicks/total_visits merged (sensitivity)."""
    if f.startswith(("clicks_", "visits_")) and not f.startswith("clicks_week"): return "act_" + f.split("_", 1)[1]
    if f in ("total_clicks", "total_visits"): return "total_activity"
    return f


COUNT_FAMS = {"vle_content", "vle_quiz_like", "vle_communication", "vle_navigation", "vle_overall", "submission", "score"}


class Gran:
    def __init__(self, feats):
        self.maps = {"feature": {f: f for f in feats}, "merged": {f: merged_of(f) for f in feats},
                     "family": {f: family_of(f) for f in feats}}


def zx(X, feats):
    """log1p counts, z-score within X, NaN->0. Only used to define effect direction."""
    Z = {}
    for f in feats:
        if f == "region": continue
        x = X[f].astype(float).values
        if f.startswith(COUNT_PREFIX): x = np.log1p(np.clip(x, 0, None))
        s = np.nanstd(x)
        Z[f] = np.nan_to_num((x - np.nanmean(x)) / s) if s > 0 else np.zeros_like(x)
    return pd.DataFrame(Z, index=X.index)


def summarize(S, Zt, G):
    """Importance = mean|group-summed SHAP|; direction = sign Spearman(group SHAP, sum of z-scored members).
    At family level direction only defined for count-type families (SELF-DESIGNED)."""
    out = {}
    for g, mp in G.maps.items():
        keys = sorted(set(mp.values()))
        Sg = S.T.groupby(S.columns.map(mp)).sum().T[keys]
        imp = Sg.abs().mean()
        cols = [k for k in keys if k != "region" and not (g == "family" and k not in COUNT_FAMS)]
        Vg = Zt.T.groupby(Zt.columns.map(mp)).sum().T
        d = pd.Series(np.nan, index=keys)
        if cols:
            rs = rankdata(Sg[cols].values, axis=0); rv = rankdata(Vg.reindex(columns=cols).fillna(0).values, axis=0)
            rs = rs - rs.mean(0); rv = rv - rv.mean(0)
            den = np.sqrt((rs ** 2).sum(0) * (rv ** 2).sum(0))
            d[cols] = np.sign(np.where(den > 0, (rs * rv).sum(0) / np.where(den > 0, den, 1), 0))
        out[g] = (imp, d)
    return out


def sim(a, b, gran):
    ia, da = a; ib, db = b
    keep = (ia > 0) | (ib > 0)
    rho = spearmanr(ia[keep], ib[keep])[0]; tau = kendalltau(ia[keep], ib[keep])[0]
    k = 3 if gran == "family" else 5
    ta, tb = set(ia.nlargest(k).index), set(ib.nlargest(k).index)
    kk = 9 if gran == "family" else 10
    u = [f for f in set(ia.nlargest(kk).index) | set(ib.nlargest(kk).index)
         if np.isfinite(da.get(f, np.nan)) and np.isfinite(db.get(f, np.nan)) and da[f] != 0 and db[f] != 0]
    return dict(spearman=rho, kendall=tau, topk=len(ta & tb) / len(ta | tb),
                sign=float(np.mean([da[f] == db[f] for f in u])) if u else np.nan)


def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p); idx = np.minimum((p * bins).astype(int), bins - 1); e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def _logit(q):
    q = np.clip(q, 1e-6, 1 - 1e-6); return np.log(q / (1 - q))


def platt(src, feats, kind):
    """Platt scaling learned entirely within the source (Lim 2026), on student-grouped 5-fold OOF predictions."""
    p = np.zeros(len(src))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(src, src.y, src.id_student):
        p[te] = Model(kind, feats).fit(src.iloc[tr], src.y.iloc[tr]).predict_proba(src.iloc[te])
    cal = LogisticRegression(C=1e6).fit(_logit(p)[:, None], src.y)
    return lambda q: cal.predict_proba(_logit(q)[:, None])[:, 1]


def psi(a, b, bins=10):
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0: return np.nan
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2: return 0.0
    ca = np.histogram(np.clip(a, edges[0], edges[-1]), edges)[0] / len(a)
    cb = np.histogram(np.clip(b, edges[0], edges[-1]), edges)[0] / len(b)
    ca, cb = np.clip(ca, 1e-4, None), np.clip(cb, 1e-4, None)
    return float(np.sum((cb - ca) * np.log(cb / ca)))


def shift_metrics(src, tgt, feats):
    rows = []
    for f in feats:
        if f == "region": continue
        a = src[f].astype(float).values; b = tgt[f].astype(float).values
        aa, bb = a[~np.isnan(a)], b[~np.isnan(b)]
        mode_b = pd.Series(bb).value_counts(normalize=True).iloc[0] if len(bb) else 1.0
        miss_b = np.mean(np.isnan(b))
        rows.append(dict(feature=f, ks=ks_2samp(aa, bb).statistic if len(aa) and len(bb) else 1.0,
                         psi=psi(a, b), zero_src=np.mean(a == 0), zero_tgt=np.mean(b == 0),
                         miss_src=np.mean(np.isnan(a)), miss_tgt=miss_b,
                         sd_ratio=(np.std(bb) / np.std(aa)) if len(aa) and np.std(aa) > 0 and len(bb) else 0.0,
                         mode_tgt=mode_b, miss_or_mode_tgt=miss_b + (1 - miss_b) * mode_b))
    return pd.DataFrame(rows).set_index("feature")


def boot_idx(t, rng):
    g = t.groupby("id_student").indices; ids = np.array(list(g.keys()))
    return np.concatenate([g[i] for i in rng.choice(ids, len(ids), replace=True)])


def run_pair(name, fam, src, tgt, feats, kind, B, cutoff):
    G = Gran(feats); rng = np.random.default_rng(SEED); Zt = zx(tgt, feats)
    m = Model(kind, feats).fit(src, src.y); p = m.predict_proba(tgt)
    row = dict(cutoff=cutoff, pair=name, family=fam, model=kind, n_src=len(src), n_tgt=len(tgt), prev_tgt=tgt.y.mean(),
               transfer_auc=roc_auc_score(tgt.y, p), transfer_brier=brier_score_loss(tgt.y, p), transfer_ece=ece(tgt.y, p),
               transfer_bal_acc=balanced_accuracy_score(tgt.y, p >= .5), transfer_f1=f1_score(tgt.y, p >= .5))
    pc = platt(src, feats, kind)(p)
    row.update(transfer_brier_platt=brier_score_loss(tgt.y, pc), transfer_ece_platt=ece(tgt.y, pc))
    tr, te = next(GroupShuffleSplit(1, test_size=.2, random_state=SEED).split(src, src.y, src.id_student))
    ms = Model(kind, feats).fit(src.iloc[tr], src.y.iloc[tr]); row["src_test_auc"] = roc_auc_score(src.y.iloc[te], ms.predict_proba(src.iloc[te]))
    p_nat, S_nat = E.oof_native(tgt, feats, kind)
    row["native_auc"] = roc_auc_score(tgt.y, p_nat); row["auc_drop"] = row["native_auc"] - row["transfer_auc"]
    row["auc_drop_vs_src"] = row["src_test_auc"] - row["transfer_auc"]
    sm_tr = summarize(m.shap(tgt), Zt, G); sm_nat = summarize(S_nat, Zt, G)
    ss = src.sample(min(len(src), 6000), random_state=SEED)
    sm_src = summarize(m.shap(ss), zx(ss, feats), G)  # label-free reference
    boots = []
    for b in range(B):  # full-size student-grouped bootstrap natives (SELF-DESIGNED noise floor)
        tb = tgt.iloc[boot_idx(tgt, rng)]
        boots.append(summarize(Model(kind, feats).fit(tb, tb.y).shap(tgt), Zt, G))
    imp_rows = []
    for g in G.maps:
        for k, v in sim(sm_tr[g], sm_nat[g], g).items(): row[f"{g}_tvn_{k}"] = v
        for k, v in sim(sm_src[g], sm_tr[g], g).items(): row[f"{g}_svt_{k}"] = v
        t_b = [sim(sm_tr[g], boots[b][g], g) for b in range(B)]
        s_b = [sim(boots[b][g], boots[(b + 1) % B][g], g) for b in range(B)]
        for k in ["spearman", "kendall", "topk", "sign"]:
            tv = np.array([d[k] for d in t_b], float); sv = np.array([d[k] for d in s_b], float); ex = sv - tv
            fin = np.isfinite(ex)
            row[f"{g}_floor_{k}"] = np.nanmean(sv); row[f"{g}_tvboot_{k}"] = np.nanmean(tv)
            row[f"{g}_excess_{k}"] = np.nanmean(ex) if fin.any() else np.nan
            lo, hi = np.percentile(ex[fin], [2.5, 97.5]) if fin.any() else (np.nan, np.nan)
            row[f"{g}_excess_{k}_lo"], row[f"{g}_excess_{k}_hi"] = lo, hi
            row[f"{g}_excess_{k}_pgt0"] = np.mean(ex[fin] > 0) if fin.any() else np.nan
        for which, smx in [("transfer_tgt", sm_tr), ("native_tgt", sm_nat), ("source_src", sm_src)]:
            for f, v in smx[g][0].items(): imp_rows.append(dict(cutoff=cutoff, model=kind, pair=name, gran=g, which=which, feature=f, value=v))
        for f, v in pd.concat([bb[g][0] for bb in boots], axis=1).mean(axis=1).items():
            imp_rows.append(dict(cutoff=cutoff, model=kind, pair=name, gran=g, which="boot_native_mean", feature=f, value=v))
    sh = shift_metrics(src, tgt, feats)
    w = sm_src["feature"][0].drop("region", errors="ignore"); w = (w / w.sum()).reindex(sh.index).fillna(0)
    row["shift_mean_ks"] = sh.ks.mean(); row["shift_wks"] = float((sh.ks * w).sum()); row["shift_wpsi"] = float((sh.psi.fillna(0) * w).sum())
    row["score_missing_tgt"] = sh.loc["mean_score", "miss_tgt"]; row["score_missing_src"] = sh.loc["mean_score", "miss_src"]
    top5 = list(sm_src["feature"][0].drop("region", errors="ignore").nlargest(5).index)
    row["src_top5"] = "|".join(top5)
    row["top5_max_degenerate"] = float(sh.loc[top5, "miss_or_mode_tgt"].max())
    row["top5_min_sd_ratio"] = float(sh.loc[top5, "sd_ratio"].min())
    row["top5_max_ks"] = float(sh.loc[top5, "ks"].max())
    # TRUST CHECK (SELF-DESIGNED; thresholds fixed a priori, not tuned): flag the transfer if any of the
    # source model's top-5 features is near-degenerate in the unlabeled target (>=90% missing or single value)
    # or its target SD falls below 25% of its source SD.
    row["trust_flag"] = int(row["top5_max_degenerate"] >= 0.90 or row["top5_min_sd_ratio"] < 0.25)
    sh = sh.reset_index(); sh.insert(0, "pair", name); sh.insert(0, "model", kind); sh.insert(0, "cutoff", cutoff)
    return row, imp_rows, sh


def main(cutoff, variant, B, jobs):
    E.XGB_PARAMS["n_jobs"] = jobs
    df, feats = load(cutoff)
    if variant == "novisits":
        feats = [f for f in feats if not f.startswith("visits_") and f != "total_visits"]
    tag = f"d{cutoff}" + ("" if variant == "base" else f"_{variant}")
    out = f"{ROOT}/results/phase2/{tag}"; os.makedirs(out, exist_ok=True)
    mods = sorted(df.code_module.unique()); pairs = []
    for mo in mods:
        s, t = df[(df.code_module == mo) & (df.year == 2013)], df[(df.code_module == mo) & (df.year == 2014)]
        if len(s) and len(t): pairs.append((f"{mo} 13->14", "cross_year_module", s, t))
    for mo in mods:
        t = df[df.code_module == mo]; s = df[(df.code_module != mo) & (~df.id_student.isin(t.id_student))]
        pairs.append((f"LOMO {mo}", "lomo", s, t))
    rows, imps, shs, pooled = [], [], [], []
    for kind in ["LR", "XGB"]:
        tr, te = next(GroupShuffleSplit(1, test_size=.2, random_state=SEED).split(df, df.y, df.id_student))
        pp = Model(kind, feats).fit(df.iloc[tr], df.y.iloc[tr]).predict_proba(df.iloc[te]); yt = df.y.iloc[te].values
        pooled.append(dict(cutoff=cutoff, model=kind, split="pooled_80_20_seed42", auc=roc_auc_score(yt, pp),
                           brier=brier_score_loss(yt, pp), ece=ece(yt, pp), n=len(df), prev=df.y.mean()))
        a, b = df[df.year == 2013], df[df.year == 2014]
        pp = Model(kind, feats).fit(a, a.y).predict_proba(b)
        pooled.append(dict(cutoff=cutoff, model=kind, split="2013->2014_pooled", auc=roc_auc_score(b.y, pp),
                           brier=brier_score_loss(b.y, pp), ece=ece(b.y.values, pp), n=len(df), prev=df.y.mean()))
        for name, fam, s, t in pairs:
            t0 = time.time()
            r, ir, sh = run_pair(name, fam, s.reset_index(drop=True), t.reset_index(drop=True), feats, kind, B, cutoff)
            r["variant"] = variant; rows.append(r); imps += ir; shs.append(sh)
            print(f"[{tag}] {kind} {name:12s} AUC tr {r['transfer_auc']:.3f} nat {r['native_auc']:.3f} "
                  f"excess fam {r['family_excess_spearman']:.2f} feat {r['feature_excess_spearman']:.2f} "
                  f"flag {r['trust_flag']} ({time.time()-t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(f"{out}/pairs.csv", index=False)
    pd.DataFrame(imps).to_csv(f"{out}/importances.csv", index=False)
    pd.concat(shs).to_csv(f"{out}/shift_by_feature.csv", index=False)
    pd.DataFrame(pooled).to_csv(f"{out}/pooled_reference.csv", index=False)
    print("DONE", tag, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, default=28); ap.add_argument("--variant", default="base")
    ap.add_argument("--B", type=int, default=30); ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args(); main(a.cutoff, a.variant, a.B, a.jobs)
