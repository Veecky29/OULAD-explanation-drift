"""Pre-deployment 'explanation trust' checks (SELF-DESIGNED). Label-free: uses only source data, the
source model's SHAP on source, and UNLABELED target features.
v1 (phase 2, absolute): flag if any source top-5 feature is >=90% missing-or-single-value in target, or
    target SD < 25% of source SD.
v2 (phase 3, source-relative; thresholds fixed a priori before evaluation): for each source top-5 feature,
    delta_deg = degeneracy_tgt - degeneracy_src, degeneracy = missing share + (1-missing) * modal-value share;
    sd_ratio = SD_tgt / SD_src (non-missing values).  Flag if any top-5 feature has delta_deg >= 0.5 or
    sd_ratio < 0.25.  Continuous score v2 = max_f max(delta_deg/0.5, log(1/sd_ratio)/log(4)) (1 = threshold).
"""
import numpy as np, pandas as pd


def _deg(x):
    miss = np.mean(np.isnan(x)); v = x[~np.isnan(x)]
    mode = pd.Series(v).value_counts(normalize=True).iloc[0] if len(v) else 1.0
    return miss + (1 - miss) * mode, miss, mode, (np.std(v) if len(v) else 0.0)


def trust_metrics(src, tgt, top5):
    rows = []
    for f in top5:
        a = src[f].astype(float).values; b = tgt[f].astype(float).values
        ds, ms, mos, sds = _deg(a); dt, mt, mot, sdt = _deg(b)
        sdr = sdt / sds if sds > 0 else (1.0 if sdt == 0 else np.inf)
        rows.append(dict(feature=f, deg_src=ds, deg_tgt=dt, delta_deg=dt - ds, sd_ratio=sdr, miss_src=ms, miss_tgt=mt))
    r = pd.DataFrame(rows)
    v1 = int((r.deg_tgt >= 0.90).any() or (r.sd_ratio < 0.25).any())
    v2 = int((r.delta_deg >= 0.5).any() or (r.sd_ratio < 0.25).any())
    score = float(np.max(np.maximum(r.delta_deg / 0.5, np.log(1 / np.clip(r.sd_ratio, 1e-3, None)) / np.log(4))))
    return dict(trust_v1=v1, trust_v2=v2, trust_v2_score=score, top5_maxdeg_tgt=float(r.deg_tgt.max()),
                top5_max_delta_deg=float(r.delta_deg.max()), top5_min_sd_ratio=float(r.sd_ratio.min()), top5="|".join(top5))
