"""Where does the model fail? OOS walk-forward predictions (single 24m model, 2023-01..2026-04)
broken down by market regime, plus staleness (accuracy vs months since training)."""
import numpy as np, pandas as pd
from btcpred import paths
paths.ensure(paths.TABLES, paths.PRED)
P = pd.read_parquet(paths.PANEL, columns=["s_close", "s_quote_volume"])
P = P[P.index < "2026-05-01"]
lc = np.log(P.s_close.astype(float)); r = lc.diff()
day_sig = r.rolling(1440).std()
reg = pd.DataFrame(index=P.index)
reg["vol_1d"] = day_sig
reg["trend_7d"] = (lc - lc.shift(7 * 1440)) / (day_sig * np.sqrt(7 * 1440))      # 7-day move in vol units
reg["vol_regime"] = np.log(day_sig / r.rolling(90 * 1440, min_periods=20 * 1440).std())  # 1d vol vs 90d
qv = P.s_quote_volume.astype(float)
reg["liq_regime"] = np.log(qv.rolling(1440).sum() / qv.rolling(90 * 1440, min_periods=20 * 1440).sum() * 90)
X = pd.read_parquet(paths.X_DEV, columns=["dow", "hour"])
T = pd.read_parquet(paths.T_DEV, columns=["zd2_1", "zd2_5"])
for h in (1, 5):
    p = pd.read_parquet(paths.PRED / f"core47_h{h}.parquet")["p"]
    z = T[f"zd2_{h}"].reindex(p.index); ok = z.notna() & (z != 0)
    p, z = p[ok], z[ok]
    D = pd.DataFrame({"hit": (p > .5) == (z > 0), "p": p, "y": (z > 0).astype(float)}).join(reg).join(X)
    D["ll"] = -(D.y * np.log(D.p) + (1 - D.y) * np.log(1 - D.p))
    base = D.groupby(D.index.to_period("M")).y.transform("mean")
    D["ll0"] = -(D.y * np.log(base) + (1 - D.y) * np.log(1 - base))
    D["skill"] = (D.ll0 - D.ll) * 1e4
    print(f"\n================ horizon {h}m (n={len(D):,}) ================")
    def show(col, bins, labels):
        g = D.groupby(pd.cut(D[col], bins, labels=labels), observed=True)
        print(f"-- {col}:", "  ".join(f"{k}: acc {v.hit.mean()*100:.2f}% skill {v.skill.mean():+.1f}bp (n={len(v)//1000}k)" for k, v in g))
    show("trend_7d", [-99, -1.5, -0.5, 0.5, 1.5, 99], ["strong bear", "bear", "flat", "bull", "strong bull"])
    show("vol_regime", [-9, -0.4, -0.1, 0.1, 0.4, 9], ["much calmer", "calmer", "normal", "wilder", "much wilder"])
    show("liq_regime", [-9, -0.4, -0.1, 0.1, 0.4, 9], ["thin", "lighter", "normal", "heavier", "heavy"])
    g = D.groupby(np.where(D.dow >= 5, "weekend", "weekday"))
    print("-- weekend:", "  ".join(f"{k}: acc {v.hit.mean()*100:.2f}% skill {v.skill.mean():+.1f}bp" for k, v in g))
    # staleness: month index within each 6-month fold (models refit at fold start)
    folds = pd.to_datetime(["2023-01-01", "2023-07-01", "2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01", "2026-01-01"], utc=True)
    start = folds[np.searchsorted(folds, D.index, side="right") - 1]
    age = ((D.index.year - start.year) * 12 + D.index.month - start.month) + 1
    g = D.groupby(age)
    print("-- model age (months since training):", "  ".join(f"m{k}: acc {v.hit.mean()*100:.2f}% skill {v.skill.mean():+.1f}bp" for k, v in g))
    # worst months
    mm = D.groupby(D.index.to_period("M")).agg(acc=("hit", "mean"), skill=("skill", "mean"))
    print("-- 5 worst months:", ", ".join(f"{k}: {v.acc*100:.1f}% / {v.skill:+.1f}bp" for k, v in mm.nsmallest(5, "skill").iterrows()))
    print("-- months with negative skill:", int((mm.skill < 0).sum()), "of", len(mm))
