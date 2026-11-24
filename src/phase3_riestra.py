"""External replication on the public Riestra-Gonzalez et al. (2021) Moodle dataset
(https://github.com/moisesriestra/moodle-early-performance-prediction; data/supervised/clean_df_{cut}_5.0.csv).
Settings and sources:
- Cutoffs 10/25/33/50% of course length: Riestra-Gonzalez et al. 2021; Angeioplastis et al. 2026 (Computers).
- Outcome: failure (mark < 5.0) = 1, i.e. 1 - BIN_TARGET of the 5.0-threshold file (BIN_TARGET rate falls
  0.74/0.59/0.20 for thresholds 2.5/5.0/8.5, so BIN_TARGET=1 means mark >= threshold): failure as positive
  class follows Angeioplastis et al. 2026.
- Unseen-course protocol: 5 course-grouped folds (StratifiedGroupKFold on COURSE, seed 42), following the
  course-grouped five-fold validation of Angeioplastis et al. 2026; each fold (~106 courses) is one target;
  source = remaining folds minus students also present in the target (student-disjoint, Lim 2026).
- Features: all 63 public supervised features (Angeioplastis et al. used 6 harmonized ones; we keep the full
  public release so feature families exist -> DEVIATION, documented). Families by event type: SELF-DESIGNED.
- Models: same LR / XGB configs as OULAD (Lim 2026-derived XGB); median imputation + standardization.
- Null control: within each target fold, random 50/50 student split (SELF-DESIGNED, as OULAD).
- B = 30 source+target bootstrap (phase-3 estimator; smaller B than OULAD's 100 for time).
Usage: python src/phase3_riestra.py   (checkpointed per job)
"""
import json, os, sys, time
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
sys.path.insert(0, os.path.dirname(__file__))
import experiment as E, phase2 as P2, phase3 as P3
from trust import trust_metrics

def _riestra_supervised_dir():
    candidates = [
        os.environ.get("RIESTRA_DATA"),
        os.path.join(os.path.dirname(__file__), "..", "external", "riestra", "data", "supervised"),
        "/workspace/external/riestra/data/supervised",
    ]
    for path in candidates:
        if path and os.path.isdir(path):
            return os.path.abspath(path)
    return os.path.abspath(candidates[1])

RD = _riestra_supervised_dir()
OUT = f"{E.ROOT}/results/phase3/riestra"; os.makedirs(OUT, exist_ok=True)
B = 30


def fam(f):
    if f.startswith("ACCOMPLISH"): return "assess_grade"
    if f.startswith(("ASSIGN_SUBMIT", "QUIZ_ATTEMPT", "QUIZ_CLOSE")): return "assess_submit_attempt"
    if f.startswith(("ASSIGN_VIEW", "QUIZ_VIEW")): return "assess_view"
    if f.startswith("COURSE_VIEW"): return "navigation"
    if f.startswith(("RESOURCE", "URL")): return "content"
    if f.startswith("FORUM"): return "communication"
    return "other"


def merged(f):  # merge time-slice variants of the same event type (collinearity sensitivity)
    for k in ["_TIME_", "_UNIQUE", "_PCT"]:
        if k in f: return f.split(k)[0]
    return f


class G:
    def __init__(self, feats):
        self.maps = {"feature": {f: f for f in feats}, "merged": {f: merged(f) for f in feats}, "family": {f: fam(f) for f in feats}}


class RModel:
    def __init__(self, kind, feats): self.kind, self.feats = kind, feats
    def fit(self, X, y):
        self.pre = Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)), ("sc", StandardScaler())]).fit(X[self.feats])
        Z = self.pre.transform(X[self.feats])
        self.m = (LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000) if self.kind == "LR" else xgb.XGBClassifier(**E.XGB_PARAMS))
        self.m.fit(Z, y); return self
    def predict_proba(self, X): return self.m.predict_proba(self.pre.transform(X[self.feats]))[:, 1]
    def shap(self, X):
        Z = self.pre.transform(X[self.feats])
        S = Z * self.m.coef_[0] if self.kind == "LR" else self.m.get_booster().predict(xgb.DMatrix(Z), pred_contribs=True)[:, :-1]
        return pd.DataFrame(S, columns=self.feats, index=X.index)


def load(cut):
    d = pd.read_csv(f"{RD}/clean_df_{cut}_5.0.csv", sep=";")
    d["y"] = (~d.BIN_TARGET.astype(bool)).astype(int); d = d.rename(columns={"UID": "id_student"})
    feats = [c for c in d.columns if c not in ("id_student", "COURSE", "BIN_TARGET", "y")]
    return d, feats


def oof_auc(t, feats, kind):
    p = np.zeros(len(t))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=E.SEED).split(t, t.y, t.id_student):
        p[te] = RModel(kind, feats).fit(t.iloc[tr], t.y.iloc[tr]).predict_proba(t.iloc[te])
    return roc_auc_score(t.y, p)


if __name__ == "__main__":
    # Usage: python src/phase3_riestra.py [CUT MODEL FOLD KIND]   (no args = run all jobs sequentially)
    E.XGB_PARAMS["n_jobs"] = 2
    P2.COUNT_FAMS.update({"assess_grade", "assess_submit_attempt", "assess_view", "navigation", "content", "communication"})
    job = sys.argv[1:]
    for cut in ([int(job[0])] if job else [10, 25, 33, 50]):
        d, feats = load(cut)
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=E.SEED).split(d, d.y, d.COURSE))
        for kind_m in ([job[1]] if job else ["XGB", "LR"]):
            for k, (_, te) in enumerate(folds):
                if job and k != int(job[2]): continue
                for kind in ([job[3]] if job else ["real", "null"]):
                    fn = f"{OUT}/c{cut}_{kind}_{kind_m}_fold{k}.json"
                    if os.path.exists(fn): continue
                    t0 = time.time(); t = d.iloc[te].reset_index(drop=True)
                    if kind == "real":
                        s = d.drop(index=d.index[te]); s = s[~s.id_student.isin(t.id_student)].reset_index(drop=True)
                    else:
                        i, j = next(GroupShuffleSplit(1, test_size=.5, random_state=E.SEED).split(t, t.y, t.id_student))
                        s, t = t.iloc[i].reset_index(drop=True), t.iloc[j].reset_index(drop=True)
                    r = P3.run(s, t, feats, kind_m, B, model_cls=RModel, gran=G(feats))
                    r["native_auc"] = oof_auc(t, feats, kind_m); r["auc_drop"] = r["native_auc"] - r["transfer_auc"]
                    mod = RModel(kind_m, feats).fit(s, s.y); ss = s.sample(min(len(s), 6000), random_state=E.SEED)
                    r.update(trust_metrics(s, t, list(mod.shap(ss).abs().mean().nlargest(5).index)))
                    r.update(cutoff=cut, model=kind_m, kind=kind, fold=k, n_courses_tgt=int(t.COURSE.nunique()), prev_tgt=float(t.y.mean()), seconds=time.time() - t0)
                    json.dump(r, open(fn + ".tmp", "w")); os.replace(fn + ".tmp", fn)
                    print(f"riestra c{cut} {kind_m} {kind} fold{k} AUC {r['transfer_auc']:.3f} nat {r['native_auc']:.3f} fam {r['family_boot_excess_spearman']:.3f} feat {r['feature_boot_excess_spearman']:.3f} ({time.time()-t0:.0f}s)", flush=True)
