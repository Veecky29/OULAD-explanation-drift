"""Print markdown tables for the phase-3 results.md section from results/phase3/*.csv (no manual transcription)."""
import os, pandas as pd
P3 = os.path.join(os.path.dirname(__file__), "..", "results", "phase3")
f = lambda x: "" if pd.isna(x) else (f"{x:.3f}" if isinstance(x, float) else str(x))
def md(df): return "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n" + "\n".join("| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False))
FAM = {"lomo": "LOMO", "cross_year_module": "跨年"}
A = pd.read_csv(f"{P3}/null_corrected_B30_all_cutoffs.csv"); A = A[A.gran != "merged"].copy(); A["family"] = A.family.map(FAM)
print("### A30\n" + md(A[["cutoff", "model", "family", "gran", "real_mean", "null_mean", "corrected", "n_real_above_null2sd", "mwu_p", "bh_q"]]))
B = pd.read_csv(f"{P3}/null_corrected_B100.csv"); B = B[(B.gran != "merged") & (B.variant == "boot")].copy(); B["family"] = B.family.map(FAM)
print("\n### B100\n" + md(B[["cutoff", "model", "family", "gran", "n_real", "n_null", "real_mean", "null_mean", "corrected", "mwu_p", "bh_q", "n_ci_lo_above_nullmean"]]))
P = pd.read_csv(f"{P3}/per_pair_corrected_ci_B100.csv"); P = P[P.gran != "merged"].copy(); P["family"] = P.family.map(FAM)
P["ci_gt0"] = P.lo > 0
S = P.groupby(["cutoff", "model", "family", "gran"]).agg(n=("ci_gt0", "size"), n_ci_gt0=("ci_gt0", "sum"), mean_corr=("corrected_excess", "mean")).reset_index()
print("\n### PB summary\n" + md(S))
sig = P[P.ci_gt0 & (P.gran == "family")][["cutoff", "model", "pair", "transfer_auc", "corrected_excess", "lo", "hi"]]
print("\n### PB family pairs with CI>0\n" + md(sig.sort_values(["cutoff", "model", "pair"])))
C = pd.read_csv(f"{P3}/trust_flag_eval_v1_v2.csv")
print("\n### trust\n" + md(C[["model", "label", "rule", "tp", "fp", "fn", "tn", "precision", "recall", "specificity", "bal_acc", "null_flags", "n_null"]]))
if os.path.exists(f"{P3}/riestra_null_corrected.csv"):
    D = pd.read_csv(f"{P3}/riestra_null_corrected.csv"); D = D[D.gran != "merged"]
    print("\n### riestra\n" + md(D[["cutoff", "model", "gran", "n_real", "n_null", "transfer_auc", "native_auc", "auc_drop", "real_mean", "null_mean", "corrected", "mwu_p", "bh_q", "trust_v1_real", "trust_v2_real", "trust_v1_null", "trust_v2_null"]]))
