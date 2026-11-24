"""Phase-3 aggregation: null-corrected tests (B=30 all cutoffs; B=100 source+target bootstrap where
available), trust-flag v1 vs v2 vs AUC-drop, Riestra external replication. Usage: python src/phase3_analysis.py"""
import glob, json, os
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), ".."); P2 = f"{ROOT}/results/phase2"; P3 = f"{ROOT}/results/phase3"
F = f"{ROOT}/figures/phase3"; os.makedirs(F, exist_ok=True)
CUTS = [7, 14, 28, 56, 112]; GR = ["feature", "merged", "family"]


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n); prev = 1.0
    for r, i in list(enumerate(o))[::-1]:
        prev = min(prev, p[i] * n / (r + 1)); q[i] = prev
    return q


def mwu(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return mannwhitneyu(a, b, alternative="greater").pvalue if len(a) and len(b) else np.nan

# ---------- A. B=30 null-corrected, all cutoffs ----------
real30 = pd.concat([pd.read_csv(f) for f in glob.glob(f"{P2}/d*/pairs.csv") if "novisits" not in f]); real30 = real30[real30.variant == "base"]
null30 = pd.concat([pd.read_csv(f) for f in glob.glob(f"{P2}/null_d*/pairs.csv")])
rows = []
for c in CUTS:
    for m in ["LR", "XGB"]:
        nn = null30[(null30.cutoff == c) & (null30.model == m)]
        for fam in ["lomo", "cross_year_module"]:
            rr = real30[(real30.cutoff == c) & (real30.model == m) & (real30.family == fam)]
            if not len(rr) or not len(nn): continue
            for g in GR:
                a, b = rr[f"{g}_excess_spearman"], nn[f"{g}_excess_spearman"]; thr = b.mean() + 2 * b.std()
                rows.append(dict(B=30, cutoff=c, model=m, family=fam, gran=g, n_real=len(rr), n_null=len(nn), real_mean=a.mean(),
                                 null_mean=b.mean(), null_sd=b.std(), corrected=a.mean() - b.mean(), n_real_above_null2sd=int((a > thr).sum()),
                                 mwu_p=mwu(a, b)))
A = pd.DataFrame(rows); A["bh_q"] = bh(A.mwu_p); A.to_csv(f"{P3}/null_corrected_B30_all_cutoffs.csv", index=False)

# ---------- B. B=100 source+target bootstrap ----------
js = [json.load(open(f)) for f in glob.glob(f"{P3}/b100/d*/*.json")]
Bdf = pd.DataFrame([{k: v for k, v in j.items() if k != "perb"} for j in js]) if js else pd.DataFrame()
rowsB, pairB = [], []
if len(Bdf):
    perb = {(j["cutoff"], j["model"], j["kind"], j["pair"]): j["perb"] for j in js}
    for (c, m), d in Bdf.groupby(["cutoff", "model"]):
        nn = d[d.kind == "null"]
        for fam in ["lomo", "cross_year_module"]:
            rr = d[(d.kind == "real") & (d.family == fam)]
            if not len(rr) or not len(nn): continue
            for g in GR:
                for var in ["boot", "fixed"]:
                    col = f"{g}_{var}_excess_spearman"; nm = nn[col].mean()
                    rowsB.append(dict(B=100, cutoff=c, model=m, family=fam, gran=g, variant=var, n_real=len(rr), n_null=len(nn),
                                      real_mean=rr[col].mean(), null_mean=nm, corrected=rr[col].mean() - nm, mwu_p=mwu(rr[col], nn[col]),
                                      n_ci_lo_above_nullmean=int((rr[f"{col}_lo"] > nm).sum())))
                for _, r in rr.iterrows():
                    pb = perb[(c, m, "real", r.pair)][g]; ex = np.array(pb["floor"]) - np.array(pb["boot"]) - nn[f"{g}_boot_excess_spearman"].mean()
                    ex = ex[np.isfinite(ex)]
                    pairB.append(dict(cutoff=c, model=m, family=fam, pair=r.pair, gran=g, transfer_auc=r.transfer_auc, transfer_auc_lo=r.transfer_auc_lo,
                                      transfer_auc_hi=r.transfer_auc_hi, corrected_excess=ex.mean(), lo=np.percentile(ex, 2.5), hi=np.percentile(ex, 97.5)))
    BB = pd.DataFrame(rowsB); BB["bh_q"] = bh(BB.mwu_p); BB.to_csv(f"{P3}/null_corrected_B100.csv", index=False)
    PB = pd.DataFrame(pairB); PB.to_csv(f"{P3}/per_pair_corrected_ci_B100.csv", index=False)
    Bdf.to_csv(f"{P3}/b100_pairs_summary.csv", index=False)

# ---------- C. trust flags ----------
tr = pd.read_csv(f"{P3}/trust_oulad.csv", keep_default_na=False, na_values=[""]) if os.path.exists(f"{P3}/trust_oulad.csv") else pd.DataFrame()
def cm(flag, lab):
    flag, lab = np.asarray(flag).astype(bool), np.asarray(lab).astype(bool)
    tp, fp, fn, tn = int((flag & lab).sum()), int((flag & ~lab).sum()), int((~flag & lab).sum()), int((~flag & ~lab).sum())
    se = tp / (tp + fn) if tp + fn else np.nan; sp = tn / (tn + fp) if tn + fp else np.nan
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=tp / (tp + fp) if tp + fp else np.nan, recall=se, specificity=sp, bal_acc=np.nanmean([se, sp]))

def nested(d, score, lab):
    pred = np.zeros(len(d), int); mod = d.pair.str.replace("LOMO ", "").str[:3].values
    for mo in np.unique(mod):
        trn = (mod != mo); best, bt = -1, np.inf
        for th in np.unique(d[score].values[trn]):
            b = cm(d[score].values[trn] >= th, lab[trn])["bal_acc"]
            if b > best: best, bt = b, th
        pred[mod == mo] = (d[score].values[mod == mo] >= bt).astype(int)
    return pred

rowsC = []
if len(tr):
    nullstats = A.set_index(["cutoff", "model", "gran"]) if len(A) else None
    t = tr[tr.kind == "real"].merge(real30[["cutoff", "model", "pair", "family", "auc_drop", "family_excess_spearman", "feature_excess_spearman"]], on=["cutoff", "model", "pair"])
    nst = null30.groupby(["cutoff", "model"]).agg(nf_m=("family_excess_spearman", "mean"), nf_s=("family_excess_spearman", "std"),
                                                  nfe_m=("feature_excess_spearman", "mean"), nfe_s=("feature_excess_spearman", "std")).reset_index()
    t = t.merge(nst, on=["cutoff", "model"])
    t["lab_family"] = (t.family_excess_spearman > t.nf_m + 2 * t.nf_s).astype(int)
    t["lab_feature"] = (t.feature_excess_spearman > t.nfe_m + 2 * t.nfe_s).astype(int)
    t.to_csv(f"{P3}/trust_eval_table.csv", index=False)
    nlt = tr[tr.kind == "null"].merge(null30[["cutoff", "model", "pair", "auc_drop"]], on=["cutoff", "model", "pair"], how="left")
    def full_thr(d, score, y):  # threshold tuned on ALL real pairs of the model (label-using; applied to null pairs for FA)
        best, bt = -1, np.inf
        for th in np.unique(d[score].values):
            b = cm(d[score].values >= th, y)["bal_acc"]
            if b > best: best, bt = b, th
        return bt
    for m in ["LR", "XGB"]:
        d = t[t.model == m].reset_index(drop=True); nl = nlt[nlt.model == m]
        for lab in ["lab_family", "lab_feature"]:
            y = d[lab].values
            th_s, th_a = full_thr(d, "trust_v2_score", y), full_thr(d, "auc_drop", y)
            for rule, pred, nflag in [("v1 absolute (phase 2, fixed)", d.trust_v1, nl.trust_v1.values),
                                      ("v2 source-relative (fixed a priori)", d.trust_v2, nl.trust_v2.values),
                                      ("v2 score, nested-LOMO threshold", nested(d, "trust_v2_score", y), (nl.trust_v2_score.values >= th_s)),
                                      ("AUC drop, nested-LOMO threshold (needs labels)", nested(d, "auc_drop", y), (nl.auc_drop.values >= th_a))]:
                nflag = np.asarray(nflag).astype(int)
                r = cm(pred, y); r.update(model=m, label=lab, rule=rule, n=len(d), base_rate=float(y.mean()),
                                          null_flags=int(nflag.sum()), n_null=len(nflag), null_false_alarm_rate=float(nflag.mean()) if len(nflag) else np.nan)
                rowsC.append(r)
    C = pd.DataFrame(rowsC); C.to_csv(f"{P3}/trust_flag_eval_v1_v2.csv", index=False)

# ---------- D. Riestra ----------
rj = [json.load(open(f)) for f in glob.glob(f"{P3}/riestra/*.json")]
if rj:
    R = pd.DataFrame([{k: v for k, v in j.items() if k != "perb"} for j in rj]); R.to_csv(f"{P3}/riestra_pairs.csv", index=False)
    rowsD = []
    for (c, m), d in R.groupby(["cutoff", "model"]):
        rr, nn = d[d.kind == "real"], d[d.kind == "null"]
        for g in GR:
            col = f"{g}_boot_excess_spearman"
            rowsD.append(dict(cutoff=c, model=m, gran=g, n_real=len(rr), n_null=len(nn), transfer_auc=rr.transfer_auc.mean(), native_auc=rr.native_auc.mean(),
                              auc_drop=rr.auc_drop.mean(), null_transfer_auc=nn.transfer_auc.mean(), real_mean=rr[col].mean(), null_mean=nn[col].mean(),
                              corrected=rr[col].mean() - nn[col].mean(), mwu_p=mwu(rr[col], nn[col]),
                              trust_v1_real=int(rr.trust_v1.sum()), trust_v2_real=int(rr.trust_v2.sum()), trust_v1_null=int(nn.trust_v1.sum()), trust_v2_null=int(nn.trust_v2.sum())))
    D = pd.DataFrame(rowsD); D["bh_q"] = bh(D.mwu_p); D.to_csv(f"{P3}/riestra_null_corrected.csv", index=False)

# ---------- Figures ----------
fig, ax = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
for i, m in enumerate(["LR", "XGB"]):
    for j, g in enumerate(["feature", "family"]):
        a = ax[i, j]
        for fam, col in [("lomo", "C0"), ("cross_year_module", "C1")]:
            d = A[(A.model == m) & (A.gran == g) & (A.family == fam)].sort_values("cutoff")
            a.plot(d.cutoff, d.corrected, "-o", color=col, label=f"{fam} B=30")
            for _, r in d.iterrows():
                if r.mwu_p < 0.05: a.annotate("*", (r.cutoff, r.corrected), fontsize=14, color=col, ha="center")
            if len(rowsB):
                e = BB[(BB.model == m) & (BB.gran == g) & (BB.family == fam) & (BB.variant == "boot")].sort_values("cutoff")
                a.scatter(e.cutoff, e.corrected, marker="D", s=60, facecolors="none", edgecolors=col, label=f"{fam} B=100 src+tgt boot")
        a.axhline(0, color="grey", lw=.6); a.set_xscale("log"); a.set_xticks(CUTS); a.set_xticklabels(CUTS)
        a.set_title(f"{m} {g}-level: real - null mean excess (* MWU p<.05)", fontsize=9); a.legend(fontsize=7)
for a in ax[1]: a.set_xlabel("cutoff day")
plt.tight_layout(); plt.savefig(f"{F}/q1_null_corrected_over_cutoffs.png", dpi=150); plt.close()

if len(pairB):
    cs = sorted(PB.cutoff.unique())
    fig, ax = plt.subplots(len(cs), 2, figsize=(11, 4.6 * len(cs)), squeeze=False)
    for i, c in enumerate(cs):
        for j, m in enumerate(["LR", "XGB"]):
            a = ax[i, j]; d = PB[(PB.cutoff == c) & (PB.model == m) & (PB.gran == "family")].sort_values(["family", "pair"]).reset_index(drop=True)
            e = PB[(PB.cutoff == c) & (PB.model == m) & (PB.gran == "feature")].set_index("pair").loc[d.pair].reset_index()
            y = np.arange(len(d))
            a.errorbar(d.corrected_excess, y + .15, xerr=[d.corrected_excess - d.lo, d.hi - d.corrected_excess], fmt="o", color="C3", ms=4, label="family")
            a.errorbar(e.corrected_excess, y - .15, xerr=[e.corrected_excess - e.lo, e.hi - e.corrected_excess], fmt="o", color="C0", ms=4, label="feature")
            a.axvline(0, color="grey"); a.set_yticks(y); a.set_yticklabels(d.pair, fontsize=7); a.legend(fontsize=7)
            a.set_title(f"{m} day {c}: null-corrected excess, B=100 src+tgt bootstrap 95% CI", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{F}/q2_per_pair_corrected_ci_B100.png", dpi=150); plt.close()

if rowsC:
    d = C[C.label == "lab_family"]; rules = list(dict.fromkeys(d.rule)); x = np.arange(len(rules)); w = .38
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for k, m in enumerate(["LR", "XGB"]):
        s = d[d.model == m].set_index("rule").loc[rules]
        ax.bar(x + (k - .5) * w, s.bal_acc, w, label=f"{m} balanced acc")
        ax.scatter(x + (k - .5) * w, s.recall, marker="^", color="k", zorder=3); ax.scatter(x + (k - .5) * w, s.specificity, marker="v", color="r", zorder=3)
        for xi, fa in zip(x, s.null_false_alarm_rate):
            if np.isfinite(fa): ax.text(xi + (k - .5) * w, 0.03, f"null FA\n{fa:.2f}", ha="center", fontsize=7, color="white")
    ax.set_xticks(x); ax.set_xticklabels(rules, fontsize=7, rotation=10); ax.set_ylim(0, 1.05); ax.axhline(.5, color="grey", lw=.6)
    ax.set_title("Flagging null-exceeding family-level drift (OULAD, 65 pairs/model); ^ recall, v specificity", fontsize=9); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{F}/q3_trust_flag_v1_v2.png", dpi=150); plt.close()

if rj:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, m in zip(ax, ["LR", "XGB"]):
        for g, col in [("feature", "C0"), ("family", "C3")]:
            d = R[(R.model == m)]
            for kind, mk in [("real", "o"), ("null", "x")]:
                e = d[d.kind == kind].groupby("cutoff")[f"{g}_boot_excess_spearman"].agg(["mean", "std"]).reset_index()
                a.errorbar(e.cutoff + (0.6 if kind == "null" else 0), e["mean"], yerr=e["std"], fmt=mk + ("-" if kind == "real" else "--"), color=col, label=f"{g} {kind}")
        a.axhline(0, color="grey", lw=.6); a.set_xlabel("cutoff (% of course)"); a.set_ylabel("excess drift (B=30 src+tgt boot)")
        a.set_title(f"Riestra (Oviedo Moodle), {m}: unseen-course folds vs null", fontsize=9); a.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{F}/q4_riestra_real_vs_null.png", dpi=150); plt.close()

pd.set_option("display.width", 250)
print(A.round(3).to_string())
if len(rowsB): print(BB.round(3).to_string())
if rowsC: print(C.round(3).to_string())
if rj: print(D.round(3).to_string())
