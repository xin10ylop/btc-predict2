import sys, time, numpy as np, pandas as pd
from btcpred.validate import walk_forward, summarize
X = pd.read_parquet("data/X_dev.parquet"); T = pd.read_parquet("data/T_dev.parquet")
fam = pd.read_csv("data/families.csv", index_col=0).iloc[:, 0]
h = int(sys.argv[1]) if len(sys.argv) > 1 else 1
t0 = time.time()
res, pred = walk_forward(X, T, h, sub=3)
summarize(res, f"all features h={h}")
pred.to_frame("p").to_parquet(f"data/pred_base_h{h}.parquet")
print(f"{time.time()-t0:.0f}s")
