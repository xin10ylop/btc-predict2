"""Research step 0: compute every feature and every target on the development period (before
the sealed holdout, 2026-05-01) and cache them, so experiments do not recompute them.

    python research/build_cache.py      # ~2 min, writes data/cache/{X_dev,T_dev}.parquet, families.csv
"""
import time
import pandas as pd
from btcpred import paths
from btcpred.data import load_panel, load_sec_offsets
from btcpred.features import make_features, make_targets, FAMILY
from btcpred.validate import HOLDOUT_START

t0 = time.time()
P = load_panel(before=HOLDOUT_START)
X, sig = make_features(P)
T = make_targets(P, sig, horizons=(1, 2, 3, 4, 5), sec=load_sec_offsets())
paths.ensure(paths.CACHE, paths.PRED)
X.to_parquet(paths.X_DEV)
T.to_parquet(paths.T_DEV)
pd.Series(FAMILY).to_csv(paths.FAMILIES)
print(X.shape, T.shape, f"{time.time() - t0:.0f}s")
print(pd.Series(FAMILY).value_counts().to_string())
