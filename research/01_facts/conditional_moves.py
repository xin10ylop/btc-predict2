import numpy as np, pandas as pd
from btcpred import paths
P = pd.read_parquet(paths.PANEL); P = P[P.index < "2026-05-01"]
c = np.log(P["s_close"].astype("float64")); r = c.diff()
vol = r.abs().ewm(span=60).mean().shift(1); z = (r/vol).clip(-8, 8)
yr = P.index.year; m = P.index.minute
D = pd.DataFrame({"z": z, "y": yr, "m": m, "r": r*1e4})
print("mean z per year x100 for key minutes:")
t = D[D.m.isin([0,1,2,15,16,17,30,31,32,45,46,47])].groupby(["m","y"])["z"].mean().unstack()*100
print(t.round(1).to_string())
print("\nmean raw bp return per year at minute 0 and 1:")
print(D[D.m.isin([0,1,31])].groupby(["m","y"])["r"].mean().unstack().round(3).to_string())
# tie rates
print("\nzero-return rate by year:", (r==0).groupby(yr).mean().round(4).to_dict())
# conditional next-minute by current z decile
D["zn"] = D["z"].shift(-1)
D["q"] = pd.qcut(D["z"], 20, labels=False, duplicates="drop")
g = D.groupby("q").agg(zmid=("z","mean"), next=("zn","mean"), pup=("zn", lambda s: (s>0).mean()), n=("zn","size"))
print("\nnext-minute z vs current z ventile:"); print(g.round(3).to_string())
# same by year for extremes
D["ext"] = pd.cut(D["z"], [-9,-4,-2,-0.5,0.5,2,4,9])
print("\nnext z by bucket and year (x100):")
print((D.groupby(["ext","y"], observed=True)["zn"].mean().unstack()*100).round(1).to_string())
