"""Run walk-forward for named feature sets (sequentially) and save per-fold results.
usage: compare_sets.py h [sets.json]   (sets.json: {"name": [features...]}; default v1_all, v2_all)"""
import sys, json, numpy as np, pandas as pd
from btcpred.validate import walk_forward, summarize
h = int(sys.argv[1])
X = pd.read_parquet("data/X_dev_v2.parquet"); T = pd.read_parquet("data/T_dev.parquet")
fam = pd.read_csv("data/families_v2.csv", index_col=0).iloc[:, 0].to_dict()
if len(sys.argv) > 2:
    SETS = json.load(open(sys.argv[2]))
else:
    SETS = {"v1_all": [c for c in X.columns if fam[c] != "intrabar"], "v2_all": list(X.columns)}
for name, feats in SETS.items():
    r, p = walk_forward(X, T, h, sub=3, target=f"zd2_{h}", feats=feats, verbose=False,
                        eval_targets=(f"zd0_{h}", f"zd10_{h}", f"zd30_{h}"))
    summarize(r, f"{name} h={h} n={len(feats)}")
    r.to_csv(f"reports/cv_{name}_h{h}.csv", index=False)
    p.to_frame("p").to_parquet(f"data/pred_{name}_h{h}.parquet")
