"""Online recalibration: each UTC day, refit a probability temperature (logit(p) *= a) on the
previous K days of already-realised outcomes, apply it to today. Shrinks the model toward 50%
when its recent edge is gone (regime break) and sharpens it when the edge is strong."""
import sys, numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
T = pd.read_parquet("data/T_dev.parquet", columns=["zd2_1", "zd2_5"])
def nll(a, lg, y):
    q = 1 / (1 + np.exp(-a * lg)); q = np.clip(q, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
for h in (1, 5):
    p = pd.read_parquet(f"data/pred_core_v3ref_h{h}.parquet")["p"]
    z = T[f"zd2_{h}"].reindex(p.index); ok = z.notna() & (z != 0)
    p, y = p[ok].clip(1e-6, 1 - 1e-6), (z[ok] > 0).astype(float)
    lg = np.log(p / (1 - p)).values; yv = y.values
    day = p.index.floor("D"); days = day.unique()
    di = {d: np.flatnonzero(day == d) for d in days}
    for K in (1, 3, 7):
        a_t = np.ones(len(p))
        for i, d in enumerate(days):
            if i < K: continue
            rows = np.concatenate([di[days[j]] for j in range(i - K, i)])
            a = minimize_scalar(nll, bounds=(0.0, 3.0), method="bounded", args=(lg[rows], yv[rows])).x
            a_t[di[d]] = a
        q = 1 / (1 + np.exp(-a_t * lg))
        base = y.groupby(day).transform("mean").values
        ll0 = -(yv * np.log(base) + (1 - yv) * np.log(1 - base))
        s_static = (ll0 - (-(yv * np.log(p.values) + (1 - yv) * np.log(1 - p.values)))) * 1e4
        s_online = (ll0 - (-(yv * np.log(q) + (1 - yv) * np.log(1 - q)))) * 1e4
        mon = pd.DataFrame({"st": s_static, "on": s_online}, index=p.index).groupby(p.index.to_period("M")).mean()
        print(f"h={h} K={K}d: static skill {s_static.mean():+.2f}bp  online {s_online.mean():+.2f}bp | months improved {(mon.on > mon.st).sum()}/{len(mon)} "
              f"| worst month static {mon.st.min():+.1f} online {mon.on.min():+.1f} | temp range {np.percentile(a_t[a_t != 1], 5):.2f}-{np.percentile(a_t[a_t != 1], 95):.2f}", flush=True)
