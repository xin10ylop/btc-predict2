"""First walk-forward baseline (62 features, scored from the exact candle close).
    python research/04_selection/baseline.py <h>   -> data/cache/pred/base_h<h>.parquet"""
import sys, time, numpy as np, pandas as pd
from btcpred import paths
paths.ensure(paths.TABLES, paths.PRED)
from btcpred.validate import walk_forward, summarize
import json
from btcpred.data import load_dev
X, T, fam = load_dev()
X = X[json.load(open(paths.SETS / "baseline.json"))["baseline_62"]]   # the first, 62-feature model
h = int(sys.argv[1]) if len(sys.argv) > 1 else 1
t0 = time.time()
res, pred = walk_forward(X, T, h, sub=3)
summarize(res, f"all features h={h}")
pred.to_frame("p").to_parquet(paths.PRED / f"base_h{h}.parquet")
print(f"{time.time()-t0:.0f}s")
