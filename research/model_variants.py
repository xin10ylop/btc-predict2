"""Compare a few principled modelling choices (few on purpose: every extra variant tried on the
same folds is a chance to overfit the validation). Target = act 2s after close.
  A  binary, 24m window (reference)
  B  binary, 12m window
  C  binary, 36m window
  D  Huber regression on clipped z, 24m  (scored via rank metrics; probability via logistic map)
  E  average of A,B,C probabilities (temporal ensemble)
"""
import sys, json, numpy as np, pandas as pd
from btcpred.validate import walk_forward, metrics, summarize, TEST_FOLDS
X = pd.read_parquet("data/X_dev_v3.parquet"); T = pd.read_parquet("data/T_dev.parquet")
h = int(sys.argv[1]); tgt = f"zd2_{h}"
feats = json.load(open("reports/selected_features.json")) if len(sys.argv) > 2 else list(X.columns)
X = X[feats]
out = {}
for name, kw in {"A_bin24": dict(train_months=24), "B_bin12": dict(train_months=12),
                 "C_bin36": dict(train_months=36)}.items():
    r, p = walk_forward(X, T, h, sub=3, target=tgt, verbose=False, **kw)
    out[name] = (r, p); summarize(r, name)
# regression variant
z = T[tgt].clip(-5, 5)
T2 = T.copy(); T2["zc"] = z
import lightgbm as lgb
from btcpred.validate import BASE_PARAMS
rows, preds = [], []
for (a, b) in TEST_FOLDS:
    idx = X.index; zt = T[tgt].values; ok = np.isfinite(zt) & (zt != 0)
    tr = np.flatnonzero(ok & (idx >= pd.Timestamp(a, tz="UTC") - pd.DateOffset(months=24)) &
                        (idx < pd.Timestamp(a, tz="UTC") - pd.Timedelta(minutes=h + 1)))[::3]
    te = np.flatnonzero(ok & (idx >= pd.Timestamp(a, tz="UTC")) & (idx < pd.Timestamp(b, tz="UTC")))
    cut = int(len(tr) * .9)
    prm = {**BASE_PARAMS, "objective": "huber", "alpha": 1.5}
    d1 = lgb.Dataset(X.values[tr[:cut]], z.values[tr[:cut]]); d2 = lgb.Dataset(X.values[tr[cut:]], z.values[tr[cut:]], reference=d1)
    m = lgb.train(prm, d1, 400, valid_sets=[d2], callbacks=[lgb.early_stopping(60, verbose=False)])
    mu_va = m.predict(X.values[tr[cut:]], num_iteration=m.best_iteration)
    # map predicted mean to probability with a 1-D logistic fit on the validation tail
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6).fit(mu_va.reshape(-1, 1), (zt[tr[cut:]] > 0).astype(int))
    mu = m.predict(X.values[te], num_iteration=m.best_iteration)
    p = lr.predict_proba(mu.reshape(-1, 1))[:, 1]
    mm = metrics(p, (zt[te] > 0).astype(int), zt[te]); mm["fold"] = a[:7]; rows.append(mm)
    preds.append(pd.Series(p, index=idx[te]))
rD = pd.DataFrame(rows); summarize(rD, "D_huber24"); out["D_huber24"] = (rD, pd.concat(preds))
# temporal ensemble
pe = (out["A_bin24"][1] + out["B_bin12"][1] + out["C_bin36"][1]) / 3
zt = T[tgt].reindex(pe.index).values
rows = []
for (a, b) in TEST_FOLDS:
    msk = (pe.index >= a) & (pe.index < b)
    mm = metrics(pe.values[msk], (zt[msk] > 0).astype(int), zt[msk]); mm["fold"] = a[:7]; rows.append(mm)
summarize(pd.DataFrame(rows), "E_ens_ABC")
pe2 = (out["A_bin24"][1] + out["D_huber24"][1].reindex(out["A_bin24"][1].index)) / 2
rows = []
for (a, b) in TEST_FOLDS:
    msk = (pe2.index >= a) & (pe2.index < b)
    mm = metrics(pe2.values[msk], (zt[msk] > 0).astype(int), zt[msk]); mm["fold"] = a[:7]; rows.append(mm)
summarize(pd.DataFrame(rows), "F_ens_A_D")
