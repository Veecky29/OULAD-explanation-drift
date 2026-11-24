"""Pilot: transfer performance drop (H1), explanation drift (H2), association (H3).
Usage: python src/experiment.py --cutoff 28
"""
import argparse, json, os, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
import xgboost as xgb

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(__file__), "..")
SEED = 42  # Wu et al. 2026 GroupShuffleSplit seed
N_BOOT = 300

# XGBoost: fixed configuration translated from Lim (2026) HistGB (180 iters, lr 0.08, 31 leaves, L2);
# fixed across modules/cutoffs so the benchmark is not adapted to held-out targets (Lim 2026).
XGB_PARAMS = dict(n_estimators=180, learning_rate=0.08, max_leaves=31, grow_policy="lossguide",
                  max_depth=0, reg_lambda=1.0, tree_method="hist", n_jobs=8, random_state=SEED,
                  eval_metric="logloss")


def _log1p(x):
    return np.log1p(np.clip(x, 0, None))


def load(cutoff):
    df = pd.read_parquet(f"{ROOT}/data/features_d{cutoff}.parquet")
    df["year"] = df["code_presentation"].str[:4].astype(int)
    feats = [c for c in df.columns if c not in
             ["code_module", "code_presentation", "id_student", "y", "final_result", "year"]]
    return df, feats


class Model:
    """Wraps LR / XGB with train-only preprocessing (median impute, log1p counts, region one-hot)."""
    def __init__(self, kind, feats):
        self.kind, self.feats = kind, feats
        self.num = [f for f in feats if f != "region"]
        self.counts = [f for f in self.num if f.startswith(("clicks_", "visits_", "total_", "precourse",
                                                           "active_days", "n_submitted", "n_late",
                                                           "days_since"))]

    def _prep(self):
        logcols = self.counts
        other = [f for f in self.num if f not in logcols]
        return ColumnTransformer([
            ("log", Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                              ("log", FunctionTransformer(_log1p, feature_names_out="one-to-one")),
                              ("sc", StandardScaler())]), logcols),
            ("num", Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)), ("sc", StandardScaler())]), other),
            ("reg", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["region"]),
        ])

    def fit(self, X, y):
        self.pre = self._prep().fit(X[self.feats])
        Z = self.pre.transform(X[self.feats])
        self.cols = self.pre.get_feature_names_out()
        # map transformed columns back to original feature names
        self.orig = [c.split("__", 1)[1] if not c.startswith("reg__") else "region" for c in self.cols]
        if self.kind == "LR":
            self.m = LogisticRegression(C=1.0, penalty="l2", class_weight="balanced", max_iter=3000)
        else:
            self.m = xgb.XGBClassifier(**XGB_PARAMS)
        self.m.fit(Z, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(self.pre.transform(X[self.feats]))[:, 1]

    def shap(self, X):
        """Per-row SHAP aggregated to original features (region one-hots summed).
        XGB: exact TreeSHAP (pred_contribs). LR: linear SHAP coef * (z - mean_train z), mean_train z=0."""
        Z = self.pre.transform(X[self.feats])
        if self.kind == "LR":
            S = Z * self.m.coef_[0]
        else:
            S = self.m.get_booster().predict(xgb.DMatrix(Z), pred_contribs=True)[:, :-1]
        S = pd.DataFrame(S, columns=self.orig, index=X.index)
        return S.T.groupby(level=0).sum().T[self.feats]


def metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    return dict(auc=roc_auc_score(y, p), f1=f1_score(y, yhat), bal_acc=balanced_accuracy_score(y, yhat),
                brier=brier_score_loss(y, p), n=len(y), prev=float(np.mean(y)))


def boot_auc(y, p, groups, rng, n=N_BOOT):
    """Student-cluster bootstrap 95% CI for AUC (Wu et al. 2026 use student-cluster bootstrap)."""
    y, p = np.asarray(y), np.asarray(p)
    ug, inv = np.unique(groups, return_inverse=True)
    idx_by = pd.Series(np.arange(len(y))).groupby(inv).apply(list).values
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(ug), len(ug))
        ii = np.concatenate([idx_by[k] for k in pick])
        if len(np.unique(y[ii])) < 2: continue
        out.append(roc_auc_score(y[ii], p[ii]))
    return np.percentile(out, [2.5, 97.5])


def oof_native(df, feats, kind, k=5):
    """Within-target optimistic baseline: stratified student-grouped k-fold CV on the target itself.
    Returns OOF predictions and OOF SHAP."""
    p = np.zeros(len(df)); S = []
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED)
    for tr, te in cv.split(df, df["y"], df["id_student"]):
        m = Model(kind, feats).fit(df.iloc[tr], df["y"].iloc[tr])
        p[te] = m.predict_proba(df.iloc[te])
        S.append(m.shap(df.iloc[te]))
    return p, pd.concat(S).loc[df.index]


def expl_summary(S, X, feats):
    imp = S.abs().mean()
    # direction = sign of Spearman corr between feature value and its SHAP (monotone effect direction)
    d = {}
    for f in feats:
        if f == "region": d[f] = np.nan; continue
        x = X[f].values; s = S[f].values
        ok = ~np.isnan(x)
        if np.nanstd(x[ok]) == 0 or np.std(s[ok]) == 0: d[f] = 0.0; continue
        d[f] = np.sign(spearmanr(x[ok], s[ok])[0])
    return imp, pd.Series(d)


def compare(impA, dirA, impB, dirB, topdir=10):
    keep = (impA > 0) | (impB > 0)
    a, b = impA[keep], impB[keep]
    rho = spearmanr(a, b)[0]; tau = kendalltau(a, b)[0]
    j = {}
    for k in (5, 10):
        ta, tb = set(impA.nlargest(k).index), set(impB.nlargest(k).index)
        j[k] = len(ta & tb) / len(ta | tb)
    # sign agreement among the union of top-10 features of both (excluding region & zero-direction)
    u = [f for f in set(impA.nlargest(topdir).index) | set(impB.nlargest(topdir).index)
         if f != "region" and dirA[f] != 0 and dirB[f] != 0]
    sign = float(np.mean([dirA[f] == dirB[f] for f in u])) if u else np.nan
    return dict(spearman=rho, kendall=tau, jac5=j[5], jac10=j[10], sign_agree=sign)


def perm_imp(model, X, y, feats, rng_seed=SEED, nmax=2000):
    if len(X) > nmax:
        X = X.sample(nmax, random_state=rng_seed); y = y.loc[X.index]
    # manual permutation on original features (sklearn needs estimator interface w/ classes_)
    base = roc_auc_score(y, model.predict_proba(X))
    rs = np.random.default_rng(rng_seed); out = {}
    for f in feats:
        drops = []
        for _ in range(3):
            Xp = X.copy(); Xp[f] = rs.permutation(Xp[f].values)
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)))
        out[f] = np.mean(drops)
    return pd.Series(out)


def run_pair(name, family, src, tgt, feats, kind, rng, do_perm=True):
    """Source model trained on 80% of source (student-grouped, seed 42); evaluated on 20% source test
    and on full target. Native = OOF grouped 5-fold CV on target. Full-source model also fit for AUC."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(gss.split(src, src["y"], src["id_student"]))
    s_tr, s_te = src.iloc[tr], src.iloc[te]
    m = Model(kind, feats).fit(s_tr, s_tr["y"])
    p_ste = m.predict_proba(s_te); p_tgt = m.predict_proba(tgt)
    mfull = Model(kind, feats).fit(src, src["y"]); p_tgt_full = mfull.predict_proba(tgt)
    p_nat, S_nat = oof_native(tgt, feats, kind)
    row = dict(pair=name, family=family, model=kind, n_src=len(src), n_tgt=len(tgt))
    for k, v in metrics(s_te["y"], p_ste).items(): row[f"src_test_{k}"] = v
    for k, v in metrics(tgt["y"], p_tgt_full).items(): row[f"transfer_{k}"] = v
    row["transfer_auc_80pct_model"] = roc_auc_score(tgt["y"], p_tgt)
    lo, hi = boot_auc(tgt["y"], p_tgt_full, tgt["id_student"].values, rng)
    row["transfer_auc_lo"], row["transfer_auc_hi"] = lo, hi
    for k, v in metrics(tgt["y"], p_nat).items(): row[f"native_{k}"] = v
    row["auc_drop"] = row["native_auc"] - row["transfer_auc"]
    row["auc_drop_rel"] = row["auc_drop"] / row["native_auc"]
    row["auc_drop_vs_src"] = row["src_test_auc"] - row["transfer_auc"]
    S_ste, S_tgt = m.shap(s_te), m.shap(tgt)
    iA, dA = expl_summary(S_ste, s_te, feats)
    iB, dB = expl_summary(S_tgt, tgt, feats)
    iN, dN = expl_summary(S_nat, tgt, feats)
    for k, v in compare(iA, dA, iB, dB).items(): row[f"srcVStgt_{k}"] = v     # same model, data shift
    for k, v in compare(iB, dB, iN, dN).items(): row[f"tgtVSnative_{k}"] = v  # transferred vs native expl.
    for k, v in compare(iA, dA, iN, dN).items(): row[f"srcVSnative_{k}"] = v
    imps = {"src_test": iA, "transfer_tgt": iB, "native_tgt": iN}
    if do_perm:
        pA = perm_imp(m, s_te, s_te["y"], feats); pB = perm_imp(m, tgt, tgt["y"], feats)
        mn = Model(kind, feats).fit(tgt, tgt["y"]); pN = perm_imp(mn, tgt, tgt["y"], feats)
        row["perm_srcVStgt_spearman"] = spearmanr(pA, pB)[0]
        row["perm_tgtVSnative_spearman"] = spearmanr(pB, pN)[0]
        row["perm_vs_shap_tgt_spearman"] = spearmanr(pB, iB)[0]  # agreement of the two importance methods
        imps.update({"perm_src_test": pA, "perm_transfer_tgt": pB, "perm_native_tgt": pN})
    return row, imps


def main(cutoff, do_perm=True):
    df, feats = load(cutoff)
    out = f"{ROOT}/results/d{cutoff}"; os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    mods = sorted(df["code_module"].unique())
    rows_wc, rows_pool, rows_pair, imp_store = [], [], [], {}
    for kind in ["LR", "XGB"]:
        print(f"== {kind} cutoff {cutoff}")
        # (a1) within-course random: per module-presentation stratified grouped 5-fold CV (optimistic)
        for (mo, pr), g in df.groupby(["code_module", "code_presentation"]):
            g = g.reset_index(drop=True)
            p, _ = oof_native(g, feats, kind)
            r = metrics(g["y"], p); r.update(module=mo, presentation=pr, model=kind); rows_wc.append(r)
        # (a2) pooled random: Wu et al. 2026 80/20 GroupShuffleSplit, 10 seeds (Table 12 robustness)
        for s in range(10):
            seed = SEED if s == 0 else s
            tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(df, df["y"], df["id_student"]))
            m = Model(kind, feats).fit(df.iloc[tr], df["y"].iloc[tr])
            r = metrics(df["y"].iloc[te], m.predict_proba(df.iloc[te])); r.update(model=kind, seed=seed)
            rows_pool.append(r)
        pairs = []
        # (b) cross-year 2013 -> 2014 pooled (Wu et al. 2026)
        pairs.append(("Y13->Y14 pooled", "cross_year_pooled", df[df.year == 2013], df[df.year == 2014]))
        # (b) cross-year per module, 2013 -> 2014 (module-wise version; CCC has no 2013 presentation)
        for mo in mods:
            s, t = df[(df.code_module == mo) & (df.year == 2013)], df[(df.code_module == mo) & (df.year == 2014)]
            if len(s) and len(t): pairs.append((f"{mo} 13->14", "cross_year_module", s, t))
        # (b') Zhang & Ahn 2023: test semester 2014J per course; SSD = all courses of 2013J; SCD = same course,
        # all previous semesters (2013B/2013J/2014B)
        for mo in mods:
            t = df[(df.code_module == mo) & (df.code_presentation == "2014J")]
            if not len(t): continue
            pairs.append((f"{mo} SSD(2013J)->2014J", "zhang_SSD", df[df.code_presentation == "2013J"], t))
            s = df[(df.code_module == mo) & (df.code_presentation != "2014J")]
            if len(s): pairs.append((f"{mo} SCD->2014J", "zhang_SCD", s, t))
        # (c) leave-one-module-out, student-disjoint: drop from training any student in the test module
        # (Lim 2026 student-disjoint LOMO; Wu et al. 2026 LOMO with overlap-exclusion sensitivity)
        for mo in mods:
            t = df[df.code_module == mo]
            s = df[(df.code_module != mo) & (~df.id_student.isin(t.id_student))]
            pairs.append((f"LOMO {mo}", "lomo", s, t))
        for name, fam, s, t in pairs:
            s, t = s.reset_index(drop=True), t.reset_index(drop=True)
            row, imps = run_pair(name, fam, s, t, feats, kind, rng, do_perm=do_perm)
            rows_pair.append(row); imp_store[(kind, name)] = imps
            print(f"  {name:28s} src {row['src_test_auc']:.3f} transfer {row['transfer_auc']:.3f} "
                  f"native {row['native_auc']:.3f} rho(tgt,nat) {row['tgtVSnative_spearman']:.2f}")
    pd.DataFrame(rows_wc).to_csv(f"{out}/within_presentation_cv.csv", index=False)
    pd.DataFrame(rows_pool).to_csv(f"{out}/pooled_random_10seeds.csv", index=False)
    pairs_df = pd.DataFrame(rows_pair); pairs_df.to_csv(f"{out}/transfer_pairs.csv", index=False)
    imp_rows = []
    for (kind, name), d in imp_store.items():
        for which, ser in d.items():
            for f, v in ser.items(): imp_rows.append(dict(model=kind, pair=name, which=which, feature=f, value=v))
    pd.DataFrame(imp_rows).to_csv(f"{out}/importances_long.csv", index=False)

    # H3: association between AUC drop and explanation drift over transfer pairs
    h3 = []
    for kind in ["LR", "XGB"]:
        for fams, label in [(["lomo", "cross_year_module"], "LOMO + cross-year per module (primary)"),
                            (["lomo"], "LOMO only"),
                            (["lomo", "cross_year_module", "zhang_SSD", "zhang_SCD"], "all module-level pairs")]:
            d = pairs_df[(pairs_df.model == kind) & pairs_df.family.isin(fams)]
            for drift in ["tgtVSnative_spearman", "srcVStgt_spearman", "perm_tgtVSnative_spearman"]:
                if drift not in d: continue
                for drop in ["auc_drop", "auc_drop_vs_src"]:
                    r, p = spearmanr(d[drop], 1 - d[drift])
                    h3.append(dict(model=kind, pairs=label, n=len(d), drop=drop, drift="1-" + drift,
                                   spearman_r=r, p=p))
    pd.DataFrame(h3).to_csv(f"{out}/h3_association.csv", index=False)
    print(pd.DataFrame(h3).round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cutoff", type=int, default=28)
    ap.add_argument("--no-perm", action="store_true")
    a = ap.parse_args(); main(a.cutoff, do_perm=not a.no_perm)
