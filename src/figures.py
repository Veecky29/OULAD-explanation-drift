"""Figures + summary tables. Usage: python src/figures.py --cutoff 28"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main(c):
    R = f"{ROOT}/results/d{c}"; F = f"{ROOT}/figures"; os.makedirs(F, exist_ok=True)
    wc = pd.read_csv(f"{R}/within_presentation_cv.csv"); pool = pd.read_csv(f"{R}/pooled_random_10seeds.csv")
    pr = pd.read_csv(f"{R}/transfer_pairs.csv"); imp = pd.read_csv(f"{R}/importances_long.csv")

    # summary table: AUC per split type per model
    rows = []
    for m in ["LR", "XGB"]:
        w = wc[wc.model == m]
        rows.append(dict(model=m, split="(a1) within-presentation grouped 5-fold CV (mean over 22)",
                         auc_mean=w.auc.mean(), auc_sd=w.auc.std(), n_units=len(w), bal_acc=w.bal_acc.mean(),
                         f1=w.f1.mean(), brier=w.brier.mean()))
        p = pool[pool.model == m]
        rows.append(dict(model=m, split="(a2) pooled 80/20 student-grouped (10 seeds)", auc_mean=p.auc.mean(),
                         auc_sd=p.auc.std(), n_units=len(p), bal_acc=p.bal_acc.mean(), f1=p.f1.mean(),
                         brier=p.brier.mean()))
        for fam, lab in [("cross_year_pooled", "(b1) 2013->2014 pooled (Wu 2026)"),
                         ("cross_year_module", "(b2) 2013->2014 per module (6)"),
                         ("zhang_SSD", "(b3) Zhang&Ahn SSD 2013J->2014J per course (7)"),
                         ("zhang_SCD", "(b4) Zhang&Ahn SCD same course prev.->2014J (7)"),
                         ("lomo", "(c) LOMO student-disjoint (7)")]:
            d = pr[(pr.model == m) & (pr.family == fam)]
            rows.append(dict(model=m, split=lab, auc_mean=d.transfer_auc.mean(), auc_sd=d.transfer_auc.std(),
                             n_units=len(d), bal_acc=d.transfer_bal_acc.mean(), f1=d.transfer_f1.mean(),
                             brier=d.transfer_brier.mean(), native_auc_mean=d.native_auc.mean(),
                             auc_drop_mean=d.auc_drop.mean(), auc_drop_vs_src_mean=d.auc_drop_vs_src.mean()))
    summ = pd.DataFrame(rows); summ.to_csv(f"{R}/summary_auc_by_split.csv", index=False)

    st = pr.groupby(["model", "family"])[[c for c in pr.columns if c.startswith(("srcVStgt_", "tgtVSnative_", "srcVSnative_", "perm_"))]].mean()
    st.to_csv(f"{R}/summary_shap_stability.csv")

    # Fig1: AUC by split type
    fig, ax = plt.subplots(figsize=(10, 4.8))
    labs = summ.split.unique(); x = np.arange(len(labs)); wdt = 0.38
    for i, m in enumerate(["LR", "XGB"]):
        s = summ[summ.model == m].set_index("split").loc[labs]
        ax.bar(x + (i - .5) * wdt, s.auc_mean, wdt, yerr=s.auc_sd.fillna(0), capsize=3, label=m)
    ax.set_xticks(x); ax.set_xticklabels([l.split(")")[0] + ")" for l in labs]); ax.set_ylim(0.5, 0.9)
    ax.set_ylabel("ROC-AUC"); ax.set_title(f"AUC by split type (cutoff day {c}); bars=mean, whiskers=SD across units")
    ax.legend(); ax.text(0, -0.2, "\n".join(labs), transform=ax.transAxes, fontsize=7, va="top")
    plt.tight_layout(); plt.savefig(f"{F}/fig1_auc_by_split_d{c}.png", dpi=150, bbox_inches="tight"); plt.close()

    # Fig2: SHAP importance rank heatmap across LOMO targets (native models) for XGB
    for m in ["XGB", "LR"]:
        d = imp[(imp.model == m) & (imp.which == "native_tgt") & imp.pair.str.startswith("LOMO")]
        piv = d.pivot(index="feature", columns="pair", values="value")
        tr = imp[(imp.model == m) & (imp.which == "src_test") & (imp.pair == "Y13->Y14 pooled")].set_index("feature").value
        top = piv.rank(ascending=False).min(axis=1).nsmallest(18).index
        rk = piv.rank(ascending=False).loc[top]
        rk = rk.loc[rk.mean(axis=1).sort_values().index]
        fig, ax = plt.subplots(figsize=(7.5, 7))
        im = ax.imshow(rk.values, cmap="viridis_r", vmin=1, vmax=25, aspect="auto")
        ax.set_xticks(range(rk.shape[1])); ax.set_xticklabels([p.replace("LOMO ", "") for p in rk.columns])
        ax.set_yticks(range(rk.shape[0])); ax.set_yticklabels(rk.index, fontsize=8)
        for i in range(rk.shape[0]):
            for j in range(rk.shape[1]):
                ax.text(j, i, int(rk.values[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if rk.values[i, j] < 10 else "black")
        plt.colorbar(im, label="mean|SHAP| rank (1 = most important)")
        ax.set_title(f"{m}: native per-module SHAP ranks (cutoff day {c})")
        plt.tight_layout(); plt.savefig(f"{F}/fig2_shap_rank_heatmap_{m}_d{c}.png", dpi=150); plt.close()

    # Fig3: H3 scatter: AUC drop vs drift (1 - Spearman transfer vs native)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, m in zip(axes, ["LR", "XGB"]):
        d = pr[(pr.model == m) & pr.family.isin(["lomo", "cross_year_module"])]
        for fam, mk in [("lomo", "o"), ("cross_year_module", "s")]:
            e = d[d.family == fam]
            ax.scatter(1 - e.tgtVSnative_spearman, e.auc_drop, marker=mk, label=fam, s=45)
            for _, r in e.iterrows():
                ax.annotate(r.pair.replace("LOMO ", "").replace(" 13->14", "*"), (1 - r.tgtVSnative_spearman, r.auc_drop), fontsize=7)
        rho, p = spearmanr(1 - d.tgtVSnative_spearman, d.auc_drop)
        ax.axhline(0, color="grey", lw=.6)
        ax.set_xlabel("explanation drift = 1 - Spearman(mean|SHAP|: transferred vs native)")
        ax.set_ylabel("AUC drop = native within-target AUC - transfer AUC")
        ax.set_title(f"{m}, day {c}: Spearman r={rho:.2f}, p={p:.3f}, n={len(d)}"); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{F}/fig3_h3_drift_vs_drop_d{c}.png", dpi=150); plt.close()

    # Fig4: per-pair transfer AUC with CI vs native AUC (LOMO + cross-year module)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, m in zip(axes, ["LR", "XGB"]):
        d = pr[(pr.model == m) & pr.family.isin(["lomo", "cross_year_module", "cross_year_pooled"])].reset_index()
        y = np.arange(len(d))
        ax.errorbar(d.transfer_auc, y, xerr=[d.transfer_auc - d.transfer_auc_lo, d.transfer_auc_hi - d.transfer_auc],
                    fmt="o", label="transfer (95% student-cluster bootstrap CI)")
        ax.scatter(d.native_auc, y, marker="x", color="red", label="native within-target CV")
        ax.scatter(d.src_test_auc, y, marker="|", color="green", s=80, label="source held-out")
        ax.set_yticks(y); ax.set_yticklabels(d.pair, fontsize=7); ax.set_xlabel("ROC-AUC"); ax.set_title(f"{m}, day {c}")
        ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{F}/fig4_pair_auc_d{c}.png", dpi=150); plt.close()
    print(summ.round(3).to_string()); print(st.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cutoff", type=int, default=28); main(ap.parse_args().cutoff)
