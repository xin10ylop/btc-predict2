"""Novel intra-bar features recovered from Binance kline volume fields."""
import numpy as np, pandas as pd
P = pd.read_parquet("data/panel.parquet"); P = P[(P.index >= "2022-01-01") & (P.index < "2026-05-01")]
T = pd.read_parquet("data/T_dev.parquet").reindex(P.index)
X0 = pd.read_parquet("data/X_dev.parquet", columns=["basis_dev_60", "ret_1", "clv"]).reindex(P.index)
f = lambda c: P[c].astype("float64")
s_c, p_c = np.log(f("s_close")), np.log(f("p_close"))
sig = (s_c.diff()**2).ewm(span=60, adjust=False).mean()**0.5
def vw(pre):
    qv, v, tbq, tbv = f(pre+"quote_volume"), f(pre+"volume"), f(pre+"taker_buy_quote_volume"), f(pre+"taker_buy_volume")
    bar = np.log(qv / v.where(v > 0))
    buy = np.log(tbq / tbv.where(tbv > 0))
    sell = np.log((qv - tbq) / (v - tbv).where((v - tbv) > 0))
    return bar, buy, sell
sb, sbuy, ssell = vw("s_"); pb, pbuy, psell = vw("p_")
F = pd.DataFrame({
  "s_close_vs_vwap": (s_c - sb) / sig,
  "p_close_vs_vwap": (p_c - pb) / sig,
  "s_buy_minus_sell_vwap": (sbuy - ssell) / sig,
  "p_buy_minus_sell_vwap": (pbuy - psell) / sig,
  "basis_fresh": ((p_c - s_c) - (pb - sb)) / sig,   # close basis minus bar-average basis
  "basis_vwap_dev60": ((pb - sb) - (pb - sb).ewm(span=60, adjust=False).mean()) / sig,
})
F = F.replace([np.inf, -np.inf], np.nan)
for h in (1, 5):
    out = {}
    for d in (0, 2, 5, 10, 30):
        z = T[f"zd{d}_{h}"]; ok = z.notna() & F.notna().all(axis=1)
        out[f"d{d}"] = F[ok].rank(pct=True).corrwith(z[ok].rank(pct=True)) * 100
    print(f"\nh={h} rank IC x100"); print(pd.DataFrame(out).round(2).to_string())
# per-year stability for the 2 most novel ones at h=1,d=2 and h=5,d=2
for col in ["s_buy_minus_sell_vwap", "p_buy_minus_sell_vwap", "basis_fresh", "p_close_vs_vwap"]:
    for h in (1, 5):
        z = T[f"zd2_{h}"]; ok = z.notna() & F[col].notna()
        g = pd.DataFrame({"a": F[col][ok].rank(pct=True), "b": z[ok].rank(pct=True)}).groupby(P.index[ok].year)
        print(col, f"h={h}", {y: round(v.a.corr(v.b)*100, 2) for y, v in g})
print("\ncorrelation with existing:\n", pd.concat([F, X0], axis=1).corr().loc[F.columns, X0.columns].round(2))
