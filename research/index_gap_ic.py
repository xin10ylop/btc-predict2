import numpy as np, pandas as pd
P = pd.read_parquet("data/panel.parquet"); P = P[(P.index >= "2023-01-01") & (P.index < "2026-05-01")]
T = pd.read_parquet("data/T_dev.parquet").reindex(P.index)
s = np.log(P.s_close.astype(float)); ix = np.log(P.ix_close.astype(float)); pc = np.log(P.p_close.astype(float))
sig = (s.diff()**2).ewm(span=60, adjust=False).mean()**0.5
ew = lambda x, n: x.ewm(span=n, adjust=False).mean()
F = pd.DataFrame({
  "spot_vs_index_dev60": ((s - ix) - ew(s - ix, 60)) / sig,
  "perp_vs_index_dev60": ((pc - ix) - ew(pc - ix, 60)) / sig,
  "basis_dev_60": ((pc - s) - ew(pc - s, 60)) / sig,
  "index_ret1_minus_spot": (ix.diff() - s.diff()) / sig,
})
for h in (1, 5):
    out = {}
    for d in (0, 2, 5, 10, 30):
        z = T[f"zd{d}_{h}"]; ok = z.notna() & F.notna().all(axis=1)
        out[f"d{d}"] = F[ok].rank(pct=True).corrwith(z[ok].rank(pct=True)) * 100
    print(f"h={h}\n", pd.DataFrame(out).round(2))
print("corr between gaps:\n", F.corr().round(2))
