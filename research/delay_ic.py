"""Which features keep their predictive power when you can only act d seconds after the close?"""
import glob, numpy as np, pandas as pd
X = pd.read_parquet("data/X_dev.parquet"); T = pd.read_parquet("data/T_dev.parquet")
S = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/sec/*.parquet"))])
S = S[~S.index.duplicated()].sort_index().reindex(X.index)
P = pd.read_parquet("data/panel.parquet", columns=["s_close"]).reindex(X.index)
lc = np.log(P.s_close)
sig = (np.log(P.s_close).diff()**2).ewm(span=60, adjust=False).mean()**0.5
msk = (X.index >= "2023-01-01")
res = {}
for h in (1, 5):
    for d in (0, 2, 5, 10, 30):
        tgt = (lc.shift(-h) - np.log(S[f"px_{d}"])) / sig
        ok = msk & tgt.notna().values
        zr = tgt[ok].clip(-6, 6).rank(pct=True)
        res[(h, d)] = X[ok].rank(pct=True).corrwith(zr) * 100
R = pd.DataFrame(res)
R.columns = [f"h{h}_d{d}" for h, d in R.columns]
R = R.reindex(R["h5_d5"].abs().sort_values(ascending=False).index)
print(R.round(2).head(45).to_string())
