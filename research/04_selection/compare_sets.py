"""Walk-forward comparison of named feature sets on identical folds and targets.

    python research/04_selection/compare_sets.py <h> research/sets/<file>.json
    HALFLIFE=6 python research/04_selection/compare_sets.py ...   # recency-weighted training

Each set in the JSON ({name: [features]}) is trained and scored on the 7 dev folds, acting
2 s after the close. Per-fold metrics -> reports/cv/<name>_h<h>.csv; out-of-sample
predictions -> data/cache/pred/<name>_h<h>.parquet. Compare two sets by differencing their
per-fold `ll_skill_bp` (a set "wins" a fold when its log-loss skill is higher).
"""
import json, os, sys
from btcpred import paths
from btcpred.data import load_dev
from btcpred.validate import walk_forward, summarize

h = int(sys.argv[1])
SETS = json.load(open(sys.argv[2]))
X, T, fam = load_dev()
paths.ensure(paths.CV, paths.PRED)
hl = float(os.environ["HALFLIFE"]) if os.environ.get("HALFLIFE") else None
for name, feats in SETS.items():
    r, p = walk_forward(X, T, h, sub=3, target=f"zd2_{h}", feats=feats, verbose=False,
                        eval_targets=(f"zd0_{h}", f"zd10_{h}", f"zd30_{h}"), halflife_months=hl)
    summarize(r, f"{name} h={h} n={len(feats)}")
    r.to_csv(paths.CV / f"{name}_h{h}.csv", index=False)
    p.to_frame("p").to_parquet(paths.PRED / f"{name}_h{h}.parquet")
