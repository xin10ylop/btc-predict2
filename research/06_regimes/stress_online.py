"""Zero-fee stress test of the full regime-aware design: model trained in today's regime
(Apr-2023..Mar-2025), with regime features at h=1, plus the live loop's 7-day online
recalibration, predicting the zero-fee regime (2022-07-08..2023-03-22)."""
import json, numpy as np, pandas as pd, lightgbm as lgb
from btcpred import paths
from scipy.optimize import minimize_scalar
from btcpred.validate import BASE_PARAMS, metrics
from btcpred.model import BTCPredictor
X = pd.read_parquet(paths.X_DEV); T = pd.read_parquet(paths.T_DEV)
close = pd.read_parquet(paths.PANEL, columns=["s_close"])["s_close"].astype(float)
D = json.load(open(paths.DESIGN))
idx = X.index
for h in (1, 5):
    feats = D["hfeatures"].get(str(h), D["base"])
    z = T[f"zd2_{h}"].values; ok = np.isfinite(z) & (z != 0); y = (z > 0).astype(int)
    tr = np.flatnonzero(ok & (idx >= "2023-04-01") & (idx < "2025-04-01"))[::3]
    te_all = np.flatnonzero((idx >= "2022-06-30") & (idx < "2023-03-22"))   # +8 days warm-up for recal
    cut = int(len(tr) * .9)
    d1 = lgb.Dataset(X[feats].values[tr[:cut]], y[tr[:cut]]); d2 = lgb.Dataset(X[feats].values[tr[cut:]], y[tr[cut:]], reference=d1)
    m = lgb.train(BASE_PARAMS, d1, 400, valid_sets=[d2], callbacks=[lgb.early_stopping(60, verbose=False)])
    p = np.clip(m.predict(X[feats].values[te_all], num_iteration=m.best_iteration), 1e-6, 1 - 1e-6)
    ti = idx[te_all]; lg = np.log(p / (1 - p))
    c0 = close.reindex(ti).values; c1 = close.shift(-h).reindex(ti).values
    okp = np.isfinite(c0) & np.isfinite(c1) & (c0 != c1); yp = (c1 > c0).astype(float)
    day = ti.floor("D"); ud = day.unique(); a_t = np.ones(len(p))
    for i, dd in enumerate(ud):
        if i < 7: continue
        rows = np.flatnonzero((day >= ud[i - 7]) & (day < dd) & okp)
        f = lambda a: -np.mean(yp[rows] * np.log(np.clip(1 / (1 + np.exp(-a * lg[rows])), 1e-6, 1)) +
                               (1 - yp[rows]) * np.log(np.clip(1 - 1 / (1 + np.exp(-a * lg[rows])), 1e-6, 1)))
        a_t[day == dd] = minimize_scalar(f, bounds=(0.0, 3.0), method="bounded").x * BTCPredictor.PROXY_ADJ[h]
    q = 1 / (1 + np.exp(-a_t * lg))
    sc = (ti >= "2022-07-08") & ok[te_all]
    for lab, pp in (("static", p), ("+ online recal", q)):
        mm = metrics(pp[sc], y[te_all][sc], z[te_all][sc])
        print(f"h={h} zero-fee stress, regime-aware design {lab:15s}: acc={mm['acc']*100:.2f}% auc={mm['auc']:.4f} "
              f"llskill={mm['ll_skill_bp']:+.1f}bp top10={mm['acc_top10']*100:.2f}%  median temp {np.median(a_t[sc]):.2f}", flush=True)
