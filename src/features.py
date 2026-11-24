"""Build leakage-controlled early-warning features from OULAD (UCI #349 mirror).

Settings and their sources (see results.md):
- Target: at-risk = final_result in {Fail, Withdrawn} vs {Pass, Distinction}
  (Wu, Jia & Pham 2026; Lim 2026; Zhang & Ahn 2023 'at-risk' = dropout + fail).
- Cutoff: day 28 primary (Wu et al. 2026; Lim 2026), day 56 = week 8 secondary
  (Zhang & Ahn 2023 earliest time point).
- Exclusion: enrollments with date_unregistration < cutoff dropped (Wu et al. 2026; Lim 2026).
- Time window: all VLE / assessment records with date < cutoff, including the pre-course
  period (negative days) (Wu et al. 2026).
- VLE features: clicks and 'visits' per activity type (20 types), total clicks/visits,
  visit days (Zhang & Ahn 2023 Table 6); weekly clicks, zero-activity flag (Wu et al. 2026);
  days since last activity.
- Assessment features: submissions before cutoff, mean score of those (Wu et al. 2026),
  late submissions (Lim 2026). Exams excluded (they are end-of-course).
- Background: gender, region, highest_education, imd_band, age_band, disability,
  num_of_prev_attempts, studied_credits, date_registration (Zhang & Ahn 2023 Table 6).
Usage: python src/features.py --cutoff 28
"""
import argparse, os
import numpy as np, pandas as pd

D = os.path.join(os.path.dirname(__file__), "..", "data")
KEY = ["code_module", "code_presentation", "id_student"]

EDU = {"No Formal quals": 0, "Lower Than A Level": 1, "A Level or Equivalent": 2,
       "HE Qualification": 3, "Post Graduate Qualification": 4}
AGE = {"0-35": 0, "35-55": 1, "55<=": 2}
IMD = {f"{i*10}-{i*10+10}%": i for i in range(10)}
IMD["10-20"] = 1  # raw data has '10-20' without % for one band


def build(cutoff: int) -> pd.DataFrame:
    info = pd.read_csv(f"{D}/studentInfo.csv")
    reg = pd.read_csv(f"{D}/studentRegistration.csv", na_values="?")
    df = info.merge(reg, on=KEY, how="left")
    n0 = len(df)
    excl = df["date_unregistration"].notna() & (df["date_unregistration"] < cutoff)
    df = df[~excl].copy()
    print(f"cutoff={cutoff}: {n0} enrollments, excluded {int(excl.sum())} unregistered before cutoff -> {len(df)}")
    df["y"] = df["final_result"].isin(["Fail", "Withdrawn"]).astype(int)

    # background
    df["gender"] = (df["gender"] == "M").astype(int)
    df["disability"] = (df["disability"] == "Y").astype(int)
    df["highest_education"] = df["highest_education"].map(EDU)
    df["age_band"] = df["age_band"].map(AGE)
    df["imd_band"] = df["imd_band"].map(IMD)  # NaN kept -> imputed in pipeline

    # VLE (chunked to keep memory low)
    vle = pd.read_csv(f"{D}/vle.csv", usecols=["id_site", "activity_type"])
    parts = []
    for ch in pd.read_csv(f"{D}/studentVle.csv", chunksize=2_000_000,
                          dtype={"id_student": np.int32, "id_site": np.int32,
                                 "date": np.int16, "sum_click": np.int32}):
        parts.append(ch[ch["date"] < cutoff])
    sv = pd.concat(parts, ignore_index=True)
    del parts
    sv = sv.merge(vle, on="id_site", how="left")
    types = sorted(vle["activity_type"].unique())
    g = sv.groupby(KEY)
    agg = pd.DataFrame({
        "total_clicks": g["sum_click"].sum(),
        "total_visits": g.size(),
        "active_days": g["date"].nunique(),
        "last_date": g["date"].max(),
        "precourse_clicks": sv[sv["date"] < 0].groupby(KEY)["sum_click"].sum(),
    })
    pt = sv.pivot_table(index=KEY, columns="activity_type", values="sum_click", aggfunc=["sum", "count"], fill_value=0)
    clicks = pt["sum"].reindex(columns=types, fill_value=0).add_prefix("clicks_")
    visits = pt["count"].reindex(columns=types, fill_value=0).add_prefix("visits_")
    pos = sv[sv["date"] >= 0].copy()
    pos["wk"] = pos["date"] // 7 + 1
    wk = pos.pivot_table(index=KEY, columns="wk", values="sum_click", aggfunc="sum", fill_value=0)
    wk = wk.reindex(columns=range(1, cutoff // 7 + 1), fill_value=0)
    wk.columns = [f"clicks_week{c}" for c in wk.columns]
    vf = agg.join([clicks, visits, wk], how="outer")
    df = df.merge(vf.reset_index(), on=KEY, how="left")
    cnt = [c for c in vf.columns if c != "last_date"]
    df[cnt] = df[cnt].fillna(0)
    df["no_vle_activity"] = (df["total_clicks"] == 0).astype(int)
    # days since last activity; earliest log is day -25, so 'never' = cutoff + 26
    df["days_since_last"] = (cutoff - df["last_date"]).fillna(cutoff + 26)
    df = df.drop(columns="last_date")

    # assessments submitted before cutoff (non-exam)
    a = pd.read_csv(f"{D}/assessments.csv", na_values="?")
    sa = pd.read_csv(f"{D}/studentAssessment.csv", na_values="?")
    sa = sa.merge(a, on="id_assessment")
    sa = sa[(sa["assessment_type"] != "Exam") & (sa["date_submitted"] < cutoff)]
    sa["late"] = (sa["date"].notna() & (sa["date_submitted"] > sa["date"])).astype(int)
    ga = sa.groupby(KEY).agg(n_submitted=("id_assessment", "count"),
                             mean_score=("score", "mean"), n_late=("late", "sum")).reset_index()
    df = df.merge(ga, on=KEY, how="left")
    df[["n_submitted", "n_late"]] = df[["n_submitted", "n_late"]].fillna(0)
    # mean_score NaN when nothing submitted -> imputed (train median) + n_submitted=0 carries the info

    keep = KEY + ["y", "final_result", "gender", "region", "highest_education", "imd_band", "age_band",
                  "disability", "num_of_prev_attempts", "studied_credits", "date_registration"] + \
        ["total_clicks", "total_visits", "active_days", "precourse_clicks", "no_vle_activity", "days_since_last"] + \
        list(clicks.columns) + list(visits.columns) + list(wk.columns) + ["n_submitted", "mean_score", "n_late"]
    return df[keep]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cutoff", type=int, default=28)
    c = ap.parse_args().cutoff
    out = build(c)
    out.to_parquet(f"{D}/features_d{c}.parquet", index=False)
    print(out.shape, "at-risk rate", round(out["y"].mean(), 4))
    print(out.groupby("code_module")["y"].agg(["size", "mean"]))
