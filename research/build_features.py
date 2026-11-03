"""Compute features + targets on the development period and cache them."""
import time, numpy as np, pandas as pd
from btcpred.features import make_features, make_targets, FAMILY
from btcpred.validate import HOLDOUT_START
t0 = time.time()
P = pd.read_parquet("data/panel.parquet")
P = P[P.index < HOLDOUT_START]
X, sig = make_features(P)
import glob
S = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/sec/*.parquet"))])
S = S[~S.index.duplicated()].sort_index()
T = make_targets(P, sig, horizons=(1, 2, 3, 4, 5), sec=S)
X.to_parquet("data/X_dev.parquet"); T.to_parquet("data/T_dev.parquet")
pd.Series(FAMILY).to_csv("data/families.csv")
print(X.shape, f"{time.time()-t0:.0f}s"); print(X.describe().T[["mean", "std", "min", "max"]].round(3).to_string())
