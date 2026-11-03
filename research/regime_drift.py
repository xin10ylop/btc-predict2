"""Do the relationships themselves drift? Monthly rank IC of the key signals (development
period only: 2021-01 .. 2026-04), targets acting 2s after the close."""
import numpy as np, pandas as pd
F = ["basis_dev_60", "basis_fresh", "p_close_vs_vwap", "eth_rel_1", "clv", "ret_1",
     "vwap_dev_30", "dist_hi_60", "ret_30", "ret_240", "vol_ratio_10_60"]
X = pd.read_parquet("data/X_dev_v3.parquet", columns=F + ["sig_bp"])
T = pd.read_parquet("data/T_dev.parquet", columns=["zd2_1", "zd2_5"])
D = X.join(T)
D = D[D.zd2_1.notna() & (D.zd2_1 != 0)]
mon = D.index.strftime("%Y-%m")
rows = []
for m, g in D.groupby(mon):
    r = g.rank(pct=True)
    rows.append({"month": m, **{f"{f}|1": r[f].corr(r["zd2_1"]) * 100 for f in F[:6]},
                 **{f"{f}|5": r[f].corr(r["zd2_5"]) * 100 for f in F}})
M = pd.DataFrame(rows).set_index("month")
M.to_csv("reports/monthly_ic.csv")
q = M.copy(); q.index = pd.PeriodIndex(q.index, freq="M")
Y = q.groupby(q.index.year).mean()
pd.set_option("display.width", 250)
print("yearly mean of monthly rank IC x100 (|1 = next 1m, |5 = next 5m):")
print(Y.round(2).T.to_string())
print("\nsign stability: share of months with the dominant sign, and worst 3-month rolling mean")
for c in M.columns:
    s = np.sign(M[c].mean())
    roll = M[c].rolling(3).mean()
    print(f"{c:22s} same-sign {np.mean(np.sign(M[c]) == s)*100:5.1f}%   mean {M[c].mean():+6.2f}   worst 3m {(roll*s).min()*s:+6.2f}  ({roll.idxmin() if s>0 else roll.idxmax()})")
