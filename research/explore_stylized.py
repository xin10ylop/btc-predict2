"""Stylized facts on the development period only (holdout 2026-05.. is sealed)."""
import numpy as np, pandas as pd
P = pd.read_parquet("data/panel.parquet")
P = P[P.index < "2026-05-01"]
c = np.log(P["s_close"].astype("float64"))
r = c.diff()
yr = P.index.year
print("std 1m ret by year (bp):", (r.groupby(yr).std()*1e4).round(2).to_dict())
print("\nACF of 1m returns (x100) by year, lags 1,2,3,5,15,30,60:")
for y in sorted(set(yr)):
    ry = r[yr == y]
    print(y, [round(ry.autocorr(l)*100, 2) for l in (1, 2, 3, 5, 15, 30, 60)])
# vol-normalized returns ACF (robust to vol clustering)
vol = r.abs().ewm(span=60).mean().shift(1)
z = (r/vol).clip(-8, 8)
print("\nACF of vol-normalized returns (x100), lags 1,2,3,5,10,15,30,60:")
for y in sorted(set(yr)):
    zy = z[yr == y]
    print(y, [round(zy.autocorr(l)*100, 2) for l in (1, 2, 3, 5, 10, 15, 30, 60)])
# 5m non-overlapping
c5 = c[P.index.minute % 5 == 0]
r5 = c5.diff()
z5 = r5/r5.abs().ewm(span=48).mean().shift(1)
print("\nACF of 5m vol-norm returns (x100), lags 1,2,3,12:", [round(z5.autocorr(l)*100, 2) for l in (1, 2, 3, 12)])
# minute of hour: absolute return and mean z
moh = P.index.minute
g = pd.DataFrame({"absr": r.abs()*1e4, "z": z, "m": moh}).groupby("m")
print("\nmean |r| bp by minute-of-hour (top/bottom):")
a = g["absr"].mean(); print(a.sort_values().tail(8).round(2).to_dict()); print(a.sort_values().head(4).round(2).to_dict())
print("\nmean z by minute (t-stat):")
mz = g["z"].mean(); sz = g["z"].std()/np.sqrt(g["z"].count())
t = (mz/sz).round(2); print(t[t.abs() > 2.5].to_dict())
# hour of day mean z
hz = pd.DataFrame({"z": z, "h": P.index.hour}).groupby("h")["z"]
th = hz.mean()/(hz.std()/np.sqrt(hz.count()))
print("\nhour-of-day mean z t-stats:", th.round(1).to_dict())
