"""Volatility (scale) model: predict E|R| for the delay-adjusted h-minute return, in units of
the running EWMA sigma. Compared against the naive "EWMA sigma is right" baseline."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
from btcpred.validate import TEST_FOLDS
X = pd.read_parquet("data/X_dev.parquet"); T = pd.read_parquet("data/T_dev.parquet")
h = int(sys.argv[1]) if len(sys.argv) > 1 else 1
z = T[f"zd2_{h}"].values
a = np.abs(z)
ok = np.isfinite(a)
P = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=3000,
         feature_fraction=0.7, bagging_fraction=0.5, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=4)
rows = []
for (s, e) in TEST_FOLDS:
    idx = X.index
    tr = np.flatnonzero(ok & (idx >= pd.Timestamp(s, tz="UTC") - pd.DateOffset(months=24)) &
                        (idx < pd.Timestamp(s, tz="UTC") - pd.Timedelta(minutes=h + 1)))[::3]
    te = np.flatnonzero(ok & (idx >= pd.Timestamp(s, tz="UTC")) & (idx < pd.Timestamp(e, tz="UTC")))
    cut = int(len(tr) * .9)
    # regress log(|z| + c): robust to fat tails; convert back via smearing on training residuals
    c = 0.05
    yt = np.log(a + c)
    d1 = lgb.Dataset(X.values[tr[:cut]], yt[tr[:cut]]); d2 = lgb.Dataset(X.values[tr[cut:]], yt[tr[cut:]], reference=d1)
    m = lgb.train(P, d1, 600, valid_sets=[d2], callbacks=[lgb.early_stopping(50, verbose=False)])
    pt = m.predict(X.values[te], num_iteration=m.best_iteration)
    # baseline: constant = training mean of log(|z|+c)
    base = np.mean(yt[tr])
    yy = yt[te]
    r2 = 1 - np.mean((yy - pt) ** 2) / np.mean((yy - base) ** 2)
    # rank corr of predicted scale with realized |z|
    ic = pd.Series(pt).corr(pd.Series(a[te]), method="spearman")
    # tail calibration: share of realized |z| above predicted 90% band
    resid_tr = yt[tr[cut:]] - m.predict(X.values[tr[cut:]], num_iteration=m.best_iteration)
    q90 = np.quantile(resid_tr, 0.90)
    cover = np.mean(yy <= pt + q90)
    rows.append(dict(fold=s[:7], r2_log=r2, spearman=ic, cover90=cover, iters=m.best_iteration))
    print(rows[-1], flush=True)
D = pd.DataFrame(rows)
print(f"h={h} mean R2(log|z|)={D.r2_log.mean():.4f} spearman={D.spearman.mean():.4f} cover90={D.cover90.mean():.4f}")
imp = pd.Series(m.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
print((imp / imp.sum()).head(15).round(3).to_string())
