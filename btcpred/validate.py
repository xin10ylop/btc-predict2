"""Walk-forward evaluation. The final holdout (>= HOLDOUT_START) is never used here."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

HOLDOUT_START = "2026-05-01"

# 6-month test folds, each trained only on data strictly before it.
TEST_FOLDS = [("2023-01-01", "2023-07-01"), ("2023-07-01", "2024-01-01"),
              ("2024-01-01", "2024-07-01"), ("2024-07-01", "2025-01-01"),
              ("2025-01-01", "2025-07-01"), ("2025-07-01", "2026-01-01"),
              ("2026-01-01", HOLDOUT_START)]

BASE_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31,
                   min_data_in_leaf=3000, feature_fraction=0.7, bagging_fraction=0.5,
                   bagging_freq=1, lambda_l2=10.0, max_bin=127, verbose=-1, num_threads=4)


def metrics(p, y, z):
    """p: P(up), y: 1/0 (ties removed), z: realized vol-normalised return."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    br = y.mean()
    ll0 = -(br * np.log(br) + (1 - br) * np.log(1 - br))
    s = p - 0.5
    top = np.abs(s) >= np.quantile(np.abs(s), 0.9)
    return dict(n=len(y), auc=roc_auc_score(y, p), acc=np.mean((p > 0.5) == y),
                ll_skill_bp=(ll0 - ll) * 1e4,  # log-loss improvement vs base rate, x1e4
                ic=np.corrcoef(s, np.clip(z, -6, 6))[0, 1],
                acc_top10=np.mean((p[top] > 0.5) == y[top]),
                edge_top10=np.mean(np.sign(s[top]) * z[top]))


def walk_forward(X, T, h, folds=TEST_FOLDS, train_months=24, params=None,
                 rounds=400, sub=2, feats=None, verbose=True, return_models=False,
                 target=None, eval_targets=()):
    """Train LightGBM per fold on [test_start - train_months, test_start - gap)."""
    params = {**BASE_PARAMS, **(params or {})}
    feats = feats or list(X.columns)
    zt = T[target or f"z{h}"].values
    ok = np.isfinite(zt) & (zt != 0)
    y_all = (zt > 0).astype("int8")
    out, preds, models = [], [], []
    for (a, b) in folds:
        ta = pd.Timestamp(a, tz="UTC") - pd.DateOffset(months=train_months)
        te = pd.Timestamp(a, tz="UTC") - pd.Timedelta(minutes=h + 1)  # purge overlap
        idx = X.index
        tr = ok & (idx >= ta) & (idx < te)
        tr_i = np.flatnonzero(tr)[::sub]
        # last 10% of training (time-ordered) for early stopping
        cut = int(len(tr_i) * 0.9)
        dtr = lgb.Dataset(X.iloc[tr_i[:cut]][feats].values, y_all[tr_i[:cut]], free_raw_data=True)
        dva = lgb.Dataset(X.iloc[tr_i[cut:]][feats].values, y_all[tr_i[cut:]], reference=dtr)
        mdl = lgb.train(params, dtr, rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(60, verbose=False)])
        te_i = np.flatnonzero(ok & (idx >= pd.Timestamp(a, tz="UTC")) & (idx < pd.Timestamp(b, tz="UTC")))
        p = mdl.predict(X.iloc[te_i][feats].values, num_iteration=mdl.best_iteration)
        m = metrics(p, y_all[te_i], zt[te_i])
        for et in eval_targets:  # same predictions scored against other (delayed) targets
            ze = T[et].values[te_i]
            okk = np.isfinite(ze) & (ze != 0)
            mm = metrics(p[okk], (ze[okk] > 0).astype(int), ze[okk])
            m[f"auc@{et}"], m[f"ic@{et}"] = mm["auc"], mm["ic"]
        m.update(fold=a[:7], iters=mdl.best_iteration)
        out.append(m)
        preds.append(pd.Series(p, index=idx[te_i]))
        if return_models:
            models.append(mdl)
        if verbose:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}, flush=True)
    res = pd.DataFrame(out)
    pred = pd.concat(preds)
    return (res, pred, models) if return_models else (res, pred)


def summarize(res, label=""):
    cols = ["auc", "acc", "ll_skill_bp", "ic", "acc_top10", "edge_top10"] + [c for c in res.columns if "@" in c]
    s = res[cols].mean()
    worst = res[cols].min()
    print(f"[{label}] mean: " + " ".join(f"{c}={s[c]:.4f}" for c in cols) +
          f" | worst-fold auc={worst['auc']:.4f} ic={worst['ic']:.4f}", flush=True)
    return s
