"""Stress test: a model trained in the CURRENT regime (Binance fees on, Apr-2023..Mar-2025)
predicting the zero-fee BTCUSDT regime (2022-07-08 .. 2023-03-22). Not a forecast claim (the
test period is earlier than training): it asks what happens if that regime comes back."""
import json, numpy as np, pandas as pd, lightgbm as lgb
from btcpred import paths
from btcpred.validate import BASE_PARAMS, metrics
X = pd.read_parquet(paths.X_DEV); T = pd.read_parquet(paths.T_DEV)
fam = pd.read_csv(paths.FAMILIES, index_col=0).iloc[:, 0].to_dict()
sel = json.load(open(paths.SELECTED))
reg = [c for c in X.columns if fam[c] == "regime"]
idx = X.index
for h in (1, 5):
    z = T[f"zd2_{h}"].values; ok = np.isfinite(z) & (z != 0); y = (z > 0).astype(int)
    tr = np.flatnonzero(ok & (idx >= "2023-04-01") & (idx < "2025-04-01"))[::3]
    te = np.flatnonzero(ok & (idx >= "2022-07-08") & (idx < "2023-03-22"))
    ctl = np.flatnonzero(ok & (idx >= "2025-04-01") & (idx < "2026-04-01"))  # same-regime control
    for name, feats in (("no regime feats", sel), ("+ regime feats", sel + reg)):
        cut = int(len(tr) * .9)
        d1 = lgb.Dataset(X[feats].values[tr[:cut]], y[tr[:cut]]); d2 = lgb.Dataset(X[feats].values[tr[cut:]], y[tr[cut:]], reference=d1)
        m = lgb.train(BASE_PARAMS, d1, 400, valid_sets=[d2], callbacks=[lgb.early_stopping(60, verbose=False)])
        for lab, rows in (("zero-fee regime 2022-07..2023-03", te), ("same-regime control 2025-04..2026-03", ctl)):
            p = m.predict(X[feats].values[rows], num_iteration=m.best_iteration)
            mm = metrics(p, y[rows], z[rows])
            print(f"h={h} {name:16s} | {lab:38s} acc={mm['acc']*100:.2f}% auc={mm['auc']:.4f} llskill={mm['ll_skill_bp']:+.1f}bp top10={mm['acc_top10']*100:.2f}%", flush=True)
