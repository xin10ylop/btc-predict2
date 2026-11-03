"""10s and 15s horizons, measured from d seconds after each candle close: how often the price
does not move at all, and which signals predict the moves that do happen (dev period)."""
import numpy as np, pandas as pd
from btcpred.data import load_dev

F = ["basis_dev_60", "basis_fresh", "basis_chg_1", "p_close_vs_vwap", "premium_dev_60", "eth_rel_1",
     "clv", "ret_1", "vwap_dev_30", "dist_hi_60", "ret_30"]
X, T, _ = load_dev(columns=F + ["sig_bp"], targets=["zs0_10", "zs2_10", "zs5_10", "zs0_15", "zs2_15", "zs5_15", "zd2_1"])
D = X.join(T)
yr = D.index.year
print("share of windows with NO price change (act +2s):")
print(pd.DataFrame({h: (D[f"zs2_{h}"] == 0).groupby(yr).mean() * 100 for h in (10, 15)}).round(1).T.to_string())
q = pd.qcut(D.sig_bp, 4, labels=["calm", "q2", "q3", "wild"])
print("by volatility quartile:", {h: ((D[f"zs2_{h}"] == 0).groupby(q, observed=True).mean() * 100).round(1).to_dict() for h in (10, 15)})

R = D[D.index >= "2023-01-01"]
out = {}
for t in ("zs0_10", "zs2_10", "zs5_10", "zs2_15", "zd2_1"):
    ok = R[t].notna() & (R[t] != 0)
    out[t] = R.loc[ok, F].rank(pct=True).corrwith(R.loc[ok, t].rank(pct=True)) * 100
print("\nrank IC x100, 2023-2026 (moves only); zs<d>_<H> = next H s acting d s after the close")
print(pd.DataFrame(out).round(2).to_string())
print("\nper-year rank IC x100 vs next 10s (act +2s):")
rows = {}
for y, g in D[D.zs2_10.notna() & (D.zs2_10 != 0)].groupby(yr[D.zs2_10.notna() & (D.zs2_10 != 0)]):
    rows[y] = g[F[:5]].rank(pct=True).corrwith(g.zs2_10.rank(pct=True)) * 100
print(pd.DataFrame(rows).round(2).to_string())
