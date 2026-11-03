"""Where do the edges concentrate? (1) lead-lag IC by volatility quartile; (2) VWAP-reversion
strength by clock position (pre-5m, pre-quarter, quarter open, funding settlement +-1 min),
and (3) its year-by-year stability. Dev period, targets acting 2 s (and 30 s) after the close."""
import numpy as np, pandas as pd
from btcpred.data import load_dev

X, T, _ = load_dev(columns=["basis_dev_60", "basis_fresh", "minute", "to_funding", "sig_bp", "vwap_dev_30"],
                   targets=["zd2_1", "zd2_5", "zd30_5"])
D = X.join(T).dropna()
ic = lambda g, f, t: g[f].rank().corr(g[t].rank()) * 100

R = D[D.index >= "2023-01-01"].copy()
R["mclass"] = np.select([R.minute % 15 == 14, R.minute % 15 == 0, R.minute % 5 == 4],
                        ["pre-quarter (:14,:29,..)", "quarter open (:00,:15,..)", "pre-5m (:04,:09,..)"], "other")
R["fund"] = np.where(R.to_funding <= 2, "settlement +-1 min", "normal")
R["vol"] = pd.qcut(R.sig_bp, 4, labels=["vol q1 (calm)", "q2", "q3", "vol q4 (wild)"])
for grp in ("vol", "mclass", "fund"):
    print(f"\n-- 2023+ by {grp}: rank IC x100 (1m basis_fresh | 1m basis_dev_60 | 5m vwap_dev_30)")
    for k, g in R.groupby(grp, observed=True):
        print(f"{str(k):28s} {ic(g, 'basis_fresh', 'zd2_1'):6.2f} {ic(g, 'basis_dev_60', 'zd2_1'):6.2f} "
              f"{ic(g, 'vwap_dev_30', 'zd2_5'):6.2f}  n={len(g):,}")

D["cls"] = np.select([D.to_funding <= 2, D.minute % 15 == 14], ["settlement", "pre-quarter"], "normal")
for t, lab in (("zd2_5", "acting 2s late"), ("zd30_5", "acting 30s late")):
    Y = D.groupby([D.index.year, "cls"]).apply(lambda g: ic(g, "vwap_dev_30", t)).unstack()
    print(f"\n-- VWAP-stretch rank IC x100 -> next 5m, {lab}, by year:"); print(Y.round(2).to_string())
