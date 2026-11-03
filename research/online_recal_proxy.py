"""Live we only see candle closes. Does fitting the online temperature on close-to-close outcomes
(zd0) instead of the act+2s outcomes (zd2) give the same benefit when scored on zd2?"""
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
T = pd.read_parquet("data/T_dev.parquet", columns=["zd0_1", "zd2_1", "zd0_5", "zd2_5"])
def nll(a, lg, y):
    q = np.clip(1 / (1 + np.exp(-a * lg)), 1e-6, 1 - 1e-6); return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
for h in (1, 5):
    p = pd.read_parquet(f"data/pred_core_v3ref_h{h}.parquet")["p"].clip(1e-6, 1 - 1e-6)
    z2 = T[f"zd2_{h}"].reindex(p.index); z0 = T[f"zd0_{h}"].reindex(p.index)
    lg = np.log(p / (1 - p)).values
    day = p.index.floor("D"); days = day.unique(); di = {d: np.flatnonzero(day == d) for d in days}
    res = {}
    for src, zz in (("fit on act+2s", z2), ("fit on closes (live proxy)", z0)):
        zv = zz.values; ok = np.isfinite(zv) & (zv != 0); yv = (zv > 0).astype(float)
        a_t = np.ones(len(p))
        for i, d in enumerate(days):
            if i < 7: continue
            rows = np.concatenate([di[days[j]] for j in range(i - 7, i)]); rows = rows[ok[rows]]
            a_t[di[d]] = minimize_scalar(nll, bounds=(0.0, 3.0), method="bounded", args=(lg[rows], yv[rows])).x
        res[src] = a_t
    zv = z2.values; ok = np.isfinite(zv) & (zv != 0); y = (zv[ok] > 0).astype(float)
    base = pd.Series(y).groupby(day[ok]).transform("mean").values
    ll0 = -(y * np.log(base) + (1 - y) * np.log(1 - base))
    for src, a_t in [("static", np.ones(len(p)))] + list(res.items()):
        q = np.clip(1 / (1 + np.exp(-a_t[ok] * lg[ok])), 1e-6, 1 - 1e-6)
        sk = (ll0 + (y * np.log(q) + (1 - y) * np.log(1 - q))).mean() * 1e4
        print(f"h={h} {src:28s} skill on act+2s outcomes {sk:+.2f}bp  median temp {np.median(a_t):.2f}")
