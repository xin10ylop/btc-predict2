"""When is the model worth listening to? Accuracy / edge as a function of confidence,
calibration, and regime breakdown. usage: selective.py pred.parquet h [delay]"""
import sys, numpy as np, pandas as pd
from btcpred import paths
path, h = sys.argv[1], int(sys.argv[2])
d = int(sys.argv[3]) if len(sys.argv) > 3 else 2
p = pd.read_parquet(path)["p"]
T = pd.read_parquet(paths.T_DEV, columns=[f"zd{d}_{h}"]).reindex(p.index)
X = pd.read_parquet(paths.X_DEV, columns=["sig_bp", "hour", "minute"]).reindex(p.index)
z = T[f"zd{d}_{h}"].values
raw_bp = z * X.sig_bp.values * np.sqrt(h)
ok = np.isfinite(z) & (z != 0)
p, z, raw_bp, X = p[ok], z[ok], raw_bp[ok], X[ok]
conf = np.abs(p.values - 0.5)
side = np.sign(p.values - 0.5)
up = z > 0
print(f"=== h={h}m, act {d}s after close, n={len(p):,} ===")
print("coverage  acc     mean_edge_bp  median_|move|_bp")
for cov in (1.0, 0.5, 0.25, 0.10, 0.05, 0.02, 0.01):
    thr = np.quantile(conf, 1 - cov)
    m = conf >= thr
    acc = np.mean((side[m] > 0) == up[m])
    print(f"{cov*100:6.0f}%   {acc*100:5.2f}%  {np.mean(side[m]*raw_bp[m]):+8.3f}      {np.median(np.abs(raw_bp[m])):7.2f}")
print("\ncalibration: predicted P(up) bin -> realised frequency")
bins = pd.cut(p.values, [0, .4, .45, .48, .5, .52, .55, .6, 1])
c = pd.DataFrame({"p": p.values, "up": up, "b": bins}).groupby("b", observed=True).agg(pred=("p", "mean"), real=("up", "mean"), n=("up", "size"))
print(c.round(4).to_string())
print("\naccuracy by vol regime (sigma terciles):")
vt = pd.qcut(X.sig_bp.values, 3, labels=["low vol", "mid vol", "high vol"])
print(pd.DataFrame({"hit": (side > 0) == up, "v": vt}).groupby("v", observed=True)["hit"].mean().round(4).to_string())
print("\naccuracy by UTC hour block:")
hb = pd.cut(X.hour.values, [-1, 5, 11, 17, 23], labels=["00-05", "06-11", "12-17", "18-23"])
print(pd.DataFrame({"hit": (side > 0) == up, "b": hb}).groupby("b", observed=True)["hit"].mean().round(4).to_string())
if h == 5:
    al = (X.minute.values % 5) == 4  # predictions made exactly at a 5-minute boundary
    print(f"\n5-minute-window aligned rows (e.g. prediction-market windows): acc={np.mean(((side > 0) == up)[al])*100:.2f}%  n={al.sum():,}")
