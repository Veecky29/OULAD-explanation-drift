"""Aggregate phase-2 outputs into tables + figures. Usage: python src/phase2_analysis.py"""
import os, glob, itertools
import numpy as np, pandas as pd
from scipy.stats import spearmanr, binomtest
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), ".."); P = f"{ROOT}/results/phase2"; F = f"{ROOT}/figures/phase2"
os.makedirs(F, exist_ok=True)
CUTS = [7, 14, 28, 56, 112]; GR = ["feature", "merged", "family"]

pairs = pd.concat([pd.read_csv(f) for f in glob.glob(f"{P}/d*/pairs.csv")], ignore_index=True)
pooled = pd.concat([pd.read_csv(f).assign(variant=os.path.basename(os.path.dirname(f))) for f in glob.glob(f"{P}/d*/pooled_reference.csv")])
base = pairs[pairs.variant == "base"].copy()
for g in GR: base[f"{g}_drift"] = 1 - base[f"{g}_tvn_spearman"]; base[f"{g}_drift_lf"] = 1 - base[f"{g}_svt_spearman"]
for g in GR: base[f"{g}_sig"] = (base[f"{g}_excess_spearman_lo"] > 0).astype(int)

# ---------- A. timeline ----------
rows = []
for (c, m), d in base.groupby(["cutoff", "model"]):
    pr = pooled[(pooled.cutoff == c) & (pooled.model == m) & (pooled.variant == f"d{c}")].set_index("split")
    for fam, e in d.groupby("family"):
        r = dict(cutoff=c, model=m, family=fam, n_pairs=len(e), pooled_random_auc=pr.loc["pooled_80_20_seed42", "auc"],
                 pooled_13to14_auc=pr.loc["2013->2014_pooled", "auc"], transfer_auc=e.transfer_auc.mean(),
                 native_auc=e.native_auc.mean(), auc_drop=e.auc_drop.mean(), auc_drop_vs_src=e.auc_drop_vs_src.mean(),
                 transfer_brier=e.transfer_brier.mean(), transfer_ece=e.transfer_ece.mean(),
                 transfer_ece_platt=e.transfer_ece_platt.mean(), n_trust_flag=int(e.trust_flag.sum()))
        for g in GR:
            r[f"{g}_tvn_spearman"] = e[f"{g}_tvn_spearman"].mean(); r[f"{g}_floor"] = e[f"{g}_floor_spearman"].mean()
            r[f"{g}_excess"] = e[f"{g}_excess_spearman"].mean(); r[f"{g}_n_ci_above0"] = int(e[f"{g}_sig"].sum())
            r[f"{g}_topk_tvn"] = e[f"{g}_tvn_topk"].mean(); r[f"{g}_sign_tvn"] = e[f"{g}_tvn_sign"].mean()
        rows.append(r)
tl = pd.DataFrame(rows).sort_values(["model", "family", "cutoff"]); tl.to_csv(f"{P}/timeline_summary.csv", index=False)

# ---------- B. per-pair CIs ----------
cols = ["cutoff", "model", "family", "pair", "n_tgt", "transfer_auc", "native_auc", "auc_drop", "auc_drop_vs_src"]
for g in GR: cols += [f"{g}_tvn_spearman", f"{g}_floor_spearman", f"{g}_excess_spearman", f"{g}_excess_spearman_lo", f"{g}_excess_spearman_hi"]
cols += ["family_tvn_topk", "family_tvn_sign", "feature_tvn_sign", "trust_flag", "top5_max_degenerate", "top5_min_sd_ratio", "src_top5"]
base[cols].to_csv(f"{P}/per_pair_excess_ci.csv", index=False)

# ---------- C. across-module tests (LOMO, 7 modules; pairs share training data -> approximate) ----------
rows = []
for (c, m), d in base[base.family == "lomo"].groupby(["cutoff", "model"]):
    for g in GR:
        x = d[f"{g}_excess_spearman"].values; k = int((x > 0).sum())
        bt = binomtest(k, len(x), 0.5, alternative="greater").pvalue
        obs = x.mean(); flips = np.array(list(itertools.product([1, -1], repeat=len(x))))
        perm = (flips * x).mean(1); pp = float(np.mean(perm >= obs - 1e-12))
        rows.append(dict(cutoff=c, model=m, gran=g, n=len(x), n_pos=k, mean_excess=obs, sign_test_p_one_sided=bt,
                         signflip_perm_p_one_sided=pp, n_ci_above0=int(d[f"{g}_sig"].sum())))
tests = pd.DataFrame(rows); tests.to_csv(f"{P}/lomo_across_module_tests.csv", index=False)

# ---------- D. what predicts drift? ----------
preds = {"auc_drop (needs target labels)": "auc_drop", "auc_drop_vs_src (needs labels)": "auc_drop_vs_src",
         "shift_wks (label-free)": "shift_wks", "shift_wpsi (label-free)": "shift_wpsi", "shift_mean_ks (label-free)": "shift_mean_ks",
         "top5_max_degenerate (label-free)": "top5_max_degenerate", "src-vs-tgt SHAP drift, same model (label-free)": None}
rows = []
for m in ["LR", "XGB"]:
    for g in ["feature", "family"]:
        for scope, d in [("all cutoffs pooled", base[base.model == m])] + [(f"cutoff {c}", base[(base.model == m) & (base.cutoff == c)]) for c in CUTS]:
            for lab, col in preds.items():
                x = d[f"{g}_drift_lf"] if col is None else d[col]
                for tgtcol in [f"{g}_drift", f"{g}_excess_spearman"]:
                    r, p = spearmanr(x, d[tgtcol], nan_policy="omit")
                    rows.append(dict(model=m, gran=g, scope=scope, n=len(d), predictor=lab, outcome=tgtcol, spearman_r=r, p=p))
pd_ = pd.DataFrame(rows); pd_.to_csv(f"{P}/drift_predictors.csv", index=False)
# within-cutoff average (reduces dependence on cutoff-level confounding)
avg = pd_[pd_.scope.str.startswith("cutoff")].groupby(["model", "gran", "predictor", "outcome"]).spearman_r.agg(["mean", "min", "max"]).reset_index()
avg.to_csv(f"{P}/drift_predictors_within_cutoff_mean.csv", index=False)

# ---------- E. trust flag evaluation ----------
def cm(flag, lab):
    flag, lab = np.asarray(flag).astype(bool), np.asarray(lab).astype(bool)
    tp = int((flag & lab).sum()); fp = int((flag & ~lab).sum()); fn = int((~flag & lab).sum()); tn = int((~flag & ~lab).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan; spec = tn / (tn + fp) if tn + fp else np.nan
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=tp / (tp + fp) if tp + fp else np.nan, recall=sens, specificity=spec,
                bal_acc=np.nanmean([sens, spec]))

rows = []
for m in ["LR", "XGB"]:
    d = base[base.model == m]
    for labname, lab in [("family excess CI>0", d.family_sig), ("feature excess CI>0", d.feature_sig),
                         ("family drift top-tercile", d.family_drift >= d.family_drift.quantile(2 / 3))]:
        r = cm(d.trust_flag, lab); r.update(model=m, label=labname, rule="fixed a-priori trust_flag", n=len(d), base_rate=float(np.mean(lab)))
        rows.append(r)
        # LOMO-CV threshold for continuous scores (SELF-DESIGNED): fit threshold on other 6 modules' pairs
        d2 = d.assign(lab=np.asarray(lab).astype(int), mod=d.pair.str[-3:].where(d.family == "lomo", d.pair.str[:3]))
        for score in ["top5_max_degenerate", "shift_wks", "family_drift_lf", "auc_drop"]:
            pred = np.zeros(len(d2), int)
            for mo in d2["mod"].unique():
                trn = d2[d2["mod"] != mo]; tst = d2["mod"] == mo
                cand = np.unique(trn[score]); best, bt = -1, None
                for th in cand:
                    b = cm(trn[score] >= th, trn.lab)["bal_acc"]
                    if b > best: best, bt = b, th
                pred[tst.values] = (d2.loc[tst, score] >= bt).astype(int)
            r = cm(pred, d2.lab); r.update(model=m, label=labname, rule=f"LOMO-CV threshold on {score}" + (" (needs labels)" if score == "auc_drop" else ""),
                                          n=len(d2), base_rate=float(d2.lab.mean())); rows.append(r)
# POST-HOC (exploratory, defined after seeing results): flag only on SD collapse relative to source
for m in ["LR", "XGB"]:
    d = base[base.model == m]
    for labname, lab in [("family excess CI>0", d.family_sig), ("feature excess CI>0", d.feature_sig)]:
        r = cm(d.top5_min_sd_ratio < 0.25, lab); r.update(model=m, label=labname, rule="POST-HOC: top-5 SD ratio < 0.25 only", n=len(d), base_rate=float(np.mean(lab))); rows.append(r)
flag = pd.DataFrame(rows); flag.to_csv(f"{P}/trust_flag_eval.csv", index=False)

# ---------- F. calibration ----------
cal = base.groupby(["model", "cutoff", "family"])[["transfer_brier", "transfer_brier_platt", "transfer_ece", "transfer_ece_platt", "prev_tgt"]].mean().reset_index()
pc = pooled[pooled.variant.str.match(r"^d\d+$")][["model", "cutoff", "split", "auc", "brier", "ece"]]
cal.to_csv(f"{P}/calibration_transfer.csv", index=False); pc.to_csv(f"{P}/calibration_pooled_reference.csv", index=False)

# ---------- G. novisits sensitivity (d28) ----------
nv = pairs[pairs.variant == "novisits"]
if len(nv):
    b28 = base[base.cutoff == 28]
    mg = b28.merge(nv, on=["model", "pair"], suffixes=("_base", "_novis"))
    keep = ["model", "pair"] + [f"{a}_{s}" for a in ["transfer_auc", "native_auc", "feature_tvn_spearman", "feature_floor_spearman",
                                                     "feature_excess_spearman", "family_excess_spearman", "trust_flag"] for s in ["base", "novis"]]
    mg[keep].to_csv(f"{P}/novisits_sensitivity_d28.csv", index=False)

# ---------- H. null control (SELF-DESIGNED): within-module random halves, no real shift ----------
nulls = pd.concat([pd.read_csv(f) for f in glob.glob(f"{P}/null_d*/pairs.csv")], ignore_index=True) if glob.glob(f"{P}/null_d*/pairs.csv") else pd.DataFrame()
if len(nulls):
    from scipy.stats import mannwhitneyu
    rows = []
    for (c, m), nn in nulls.groupby(["cutoff", "model"]):
        for fam in ["lomo", "cross_year_module"]:
            rr = base[(base.cutoff == c) & (base.model == m) & (base.family == fam)]
            for g in GR:
                a, b = rr[f"{g}_excess_spearman"], nn[f"{g}_excess_spearman"]
                rows.append(dict(cutoff=c, model=m, family=fam, gran=g, real_mean=a.mean(), null_mean=b.mean(), null_sd=b.std(),
                                 null_corrected=a.mean() - b.mean(), real_n_ci_above0=int((rr[f"{g}_excess_spearman_lo"] > 0).sum()), n_real=len(rr),
                                 null_n_ci_above0=int((nn[f"{g}_excess_spearman_lo"] > 0).sum()), n_null=len(nn),
                                 mwu_p_one_sided=mannwhitneyu(a, b, alternative="greater").pvalue,
                                 null_trust_flags=int(nn.trust_flag.sum())))
    nullcmp = pd.DataFrame(rows); nullcmp.to_csv(f"{P}/null_control_comparison.csv", index=False)
    print(nullcmp.round(3).to_string())

# ---------- Figures ----------
# P1: drift & AUC gap over cutoffs
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
for m, ls in [("LR", "--"), ("XGB", "-")]:
    for fam, col in [("lomo", "C0"), ("cross_year_module", "C1")]:
        t = tl[(tl.model == m) & (tl.family == fam)]
        ax[0].plot(t.cutoff, t.auc_drop_vs_src, ls, color=col, marker="o", label=f"{m} {fam}")
        ax[1].plot(t.cutoff, t.feature_excess, ls, color=col, marker="o", label=f"{m} {fam}")
        ax[2].plot(t.cutoff, t.family_excess, ls, color=col, marker="o", label=f"{m} {fam}")
for a, tt in zip(ax, ["AUC drop vs source held-out", "feature-level excess drift (floor - transfer Spearman)", "family-level excess drift"]):
    a.set_xscale("log"); a.set_xticks(CUTS); a.set_xticklabels(CUTS); a.axhline(0, color="grey", lw=.6); a.set_xlabel("cutoff day"); a.set_title(tt, fontsize=9)
ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{F}/p1_drift_over_cutoffs.png", dpi=150); plt.close()

# P2: family-level heatmap (XGB, d28 & d56): native vs transferred share of total |SHAP|, LOMO
imp = pd.concat([pd.read_csv(f) for f in [f"{P}/d28/importances.csv", f"{P}/d56/importances.csv"] if os.path.exists(f)])
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, c in enumerate([28, 56]):
    for j, which in enumerate(["native_tgt", "transfer_tgt"]):
        d = imp[(imp.model == "XGB") & (imp.cutoff == c) & (imp.gran == "family") & (imp.which == which) & imp.pair.str.startswith("LOMO")]
        pv = d.pivot(index="feature", columns="pair", values="value"); pv = pv / pv.sum()
        a = axes[i, j]; im = a.imshow(pv.values, cmap="magma_r", vmin=0, vmax=.5, aspect="auto")
        a.set_xticks(range(pv.shape[1])); a.set_xticklabels([p[-3:] for p in pv.columns]); a.set_yticks(range(pv.shape[0])); a.set_yticklabels(pv.index, fontsize=8)
        for r_ in range(pv.shape[0]):
            for c_ in range(pv.shape[1]): a.text(c_, r_, f"{pv.values[r_, c_]:.2f}", ha="center", va="center", fontsize=7, color="white" if pv.values[r_, c_] > .25 else "black")
        a.set_title(f"XGB day {c}: {'native (target-trained)' if which=='native_tgt' else 'transferred (LOMO)'} share of |SHAP|", fontsize=9)
plt.colorbar(im, ax=axes, shrink=.6); plt.savefig(f"{F}/p2_family_heatmap_xgb.png", dpi=150, bbox_inches="tight"); plt.close()

# P3: shift (label-free) vs drift, and AUC drop vs drift
fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
for i, m in enumerate(["LR", "XGB"]):
    d = base[base.model == m]
    for j, (x, lab) in enumerate([("top5_max_degenerate", "label-free: max degeneracy of source top-5 features in target"), ("auc_drop_vs_src", "AUC drop vs source (needs labels)")]):
        a = ax[i, j]
        for fam, mk in [("lomo", "o"), ("cross_year_module", "s")]:
            e = d[d.family == fam]; sc = a.scatter(e[x], e.family_excess_spearman, c=np.log(e.cutoff), cmap="viridis", marker=mk, label=fam, s=30)
        r, p = spearmanr(d[x], d.family_excess_spearman)
        a.axhline(0, color="grey", lw=.6); a.set_xlabel(lab, fontsize=8); a.set_ylabel("family-level excess drift")
        a.set_title(f"{m}: Spearman r={r:.2f} (n={len(d)}, pairs x 5 cutoffs; not independent)", fontsize=9); a.legend(fontsize=7)
cb = plt.colorbar(sc, ax=ax); cb.set_ticks(np.log(CUTS)); cb.set_ticklabels(CUTS); cb.set_label("cutoff day")
plt.savefig(f"{F}/p3_shift_vs_drift.png", dpi=150, bbox_inches="tight"); plt.close()

# P4: trust-flag performance
f2 = flag[flag.label == "family excess CI>0"]
fig, ax = plt.subplots(figsize=(10, 4.2)); x = np.arange(f2.rule.nunique()); w = .38
for k, m in enumerate(["LR", "XGB"]):
    s = f2[f2.model == m].set_index("rule")
    ax.bar(x + (k - .5) * w, s.bal_acc, w, label=f"{m} balanced acc")
    ax.scatter(x + (k - .5) * w, s.recall, marker="^", color="k", zorder=3); ax.scatter(x + (k - .5) * w, s.specificity, marker="v", color="r", zorder=3)
ax.set_xticks(x); ax.set_xticklabels([r.replace("LOMO-CV threshold on ", "LOMO-CV: ") for r in f2[f2.model == "LR"].rule], fontsize=7, rotation=12)
ax.axhline(.5, color="grey", lw=.6); ax.set_ylim(0, 1.05); ax.set_title("Flagging 'explanation drift beyond noise' (family excess CI>0); ^ recall, v specificity")
ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{F}/p4_trust_flag_performance.png", dpi=150); plt.close()

# P5: forest plot per-pair family excess CI (XGB & LR, d28)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
for a, m in zip(ax, ["LR", "XGB"]):
    d = base[(base.model == m) & (base.cutoff == 28)].reset_index(drop=True); y = np.arange(len(d))
    for g, off, col in [("feature", -.15, "C0"), ("family", .15, "C3")]:
        a.errorbar(d[f"{g}_excess_spearman"], y + off, xerr=[d[f"{g}_excess_spearman"] - d[f"{g}_excess_spearman_lo"], d[f"{g}_excess_spearman_hi"] - d[f"{g}_excess_spearman"]], fmt="o", color=col, label=g, ms=4)
    if len(nulls):
        nn = nulls[(nulls.model == m) & (nulls.cutoff == 28)]
        a.axvline(nn.feature_excess_spearman.mean(), color="C0", ls=":", label="null mean (feature)")
        a.axvline(nn.family_excess_spearman.mean(), color="C3", ls=":", label="null mean (family)")
    a.axvline(0, color="grey"); a.set_yticks(y); a.set_yticklabels(d.pair, fontsize=7); a.set_title(f"{m}, day 28: excess drift (bootstrap 95% CI)", fontsize=9); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{F}/p5_excess_ci_forest_d28.png", dpi=150); plt.close()

pd.set_option("display.width", 250)
print(tl[["cutoff", "model", "family", "pooled_random_auc", "pooled_13to14_auc", "transfer_auc", "native_auc", "auc_drop", "auc_drop_vs_src",
          "feature_tvn_spearman", "feature_floor", "feature_excess", "feature_n_ci_above0", "family_tvn_spearman", "family_floor", "family_excess", "family_n_ci_above0", "n_trust_flag"]].round(3).to_string())
print(tests.round(3).to_string()); print(flag.round(3).to_string())
print(avg[avg.outcome.str.endswith("excess_spearman")].round(3).to_string())
