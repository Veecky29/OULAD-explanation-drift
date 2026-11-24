"""Sanity / leakage checks. Usage: python src/leakage_checks.py"""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
sys.path.insert(0, os.path.dirname(__file__))
from experiment import load, ROOT, SEED

rows = []
for c in [28, 56]:
    df, feats = load(c)
    D = f"{ROOT}/data"
    reg = pd.read_csv(f"{D}/studentRegistration.csv", na_values="?")
    m = df.merge(reg, on=["code_module", "code_presentation", "id_student"], suffixes=("", "_r"))
    rows.append(dict(cutoff=c, check="no enrollment with date_unregistration < cutoff",
                     ok=bool(~(m.date_unregistration < c).any()), detail=""))
    rows.append(dict(cutoff=c, check="n enrollments / at-risk prevalence", ok=True,
                     detail=f"{len(df)} / {df.y.mean():.4f}; withdrawn after cutoff kept as positives: {(df.final_result=='Withdrawn').sum()}"))
    # grouped splits: student overlap
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(df, df.y, df.id_student))
    ov = len(set(df.id_student.iloc[tr]) & set(df.id_student.iloc[te]))
    rows.append(dict(cutoff=c, check="pooled 80/20 grouped split student overlap = 0", ok=ov == 0, detail=str(ov)))
    ovs = []
    for mo in df.code_module.unique():
        t = df[df.code_module == mo]; s = df[(df.code_module != mo) & (~df.id_student.isin(t.id_student))]
        ovs.append(len(set(s.id_student) & set(t.id_student)))
    rows.append(dict(cutoff=c, check="LOMO train/test student overlap = 0", ok=max(ovs) == 0, detail=str(ovs)))
    y13 = set(df[df.year == 2013].id_student); y14 = set(df[df.year == 2014].id_student)
    rows.append(dict(cutoff=c, check="cross-year: students in both 2013 and 2014 (retakers, kept as in Wu 2026)", ok=True,
                     detail=f"{len(y13 & y14)} students; {df[(df.year==2014)&df.id_student.isin(y13)].shape[0]} 2014 records"))
    # single-feature AUC screen: any feature alone with AUC > 0.85 would be suspicious
    au = {}
    for f in feats:
        if f == "region": continue
        x = df[f].fillna(df[f].median())
        if x.nunique() > 1:
            a = roc_auc_score(df.y, x); au[f] = max(a, 1 - a)
    top = sorted(au.items(), key=lambda kv: -kv[1])[:5]
    rows.append(dict(cutoff=c, check="max single-feature AUC < 0.85", ok=top[0][1] < 0.85,
                     detail="; ".join(f"{k}={v:.3f}" for k, v in top)))
    rows.append(dict(cutoff=c, check="enrollments registered on/after cutoff (not observable in deployment; kept, as in Wu 2026)",
                     ok=True, detail=str(int((df.date_registration >= c).sum()))))
    rows.append(dict(cutoff=c, check="enrollments with no submitted assessment before cutoff (mean_score imputed)", ok=True,
                     detail=f"{(df.n_submitted==0).mean():.3f} overall; by module: " +
                     ", ".join(f"{k}={v:.2f}" for k, v in df.groupby('code_module').n_submitted.apply(lambda s: (s==0).mean()).items())))
out = pd.DataFrame(rows); out.to_csv(f"{ROOT}/results/leakage_checks.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 200); print(out.to_string())
