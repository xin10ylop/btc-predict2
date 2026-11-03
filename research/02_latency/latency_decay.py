"""How much of the predicted move is still ahead of you if you act d seconds after the bar close?
Uses tick data (aggTrades) for 3 sample days and the out-of-sample predictions of the walk-forward model."""
import zipfile, numpy as np, pandas as pd
from btcpred import paths
paths.ensure(paths.TABLES, paths.PRED)
days = ["2026-02-05", "2026-03-10", "2026-04-15"]
def load(kind, d):
    z = zipfile.ZipFile(paths.TICKS / f"{kind}_BTCUSDT-aggTrades-{d}.zip")
    f = z.open(z.namelist()[0]); first = f.readline().decode(); f.close()
    df = pd.read_csv(z.open(z.namelist()[0]), header=0 if first.startswith("agg") else None)
    t = df.iloc[:, 5].astype("int64").values
    t = np.where(t > 10**14, t // 1000, t)  # us -> ms
    return pd.Series(df.iloc[:, 1].astype(float).values, index=pd.to_datetime(t, unit="ms", utc=True))
out = {}
for h in (1, 5):
    pred = pd.read_parquet(paths.PRED / f"base_h{h}.parquet")["p"]
    rows = []
    for d in days:
        px = load("spot", d)
        sec = px.groupby(px.index.floor("1s")).last()
        full = pd.date_range(f"{d} 00:00:00", f"{d} 23:59:59", freq="1s", tz="UTC").union(
               pd.date_range(f"{d} 23:59:59", periods=400, freq="1s", tz="UTC"))
        sec = np.log(sec.reindex(full).ffill())
        p = pred[(pred.index >= f"{d} 00:00") & (pred.index < f"{d} 23:50")]
        close_t = p.index + pd.Timedelta(seconds=59)  # last second of bar t
        target_t = close_t + pd.Timedelta(minutes=h)
        base = sec.reindex(close_t).values
        tgt = sec.reindex(target_t).values
        s = p.values - 0.5
        for dsec in (0, 1, 2, 3, 5, 10, 20, 30, 45):
            start = sec.reindex(close_t + pd.Timedelta(seconds=dsec)).values
            rem = tgt - start
            ok = np.isfinite(rem)
            rows.append((d, dsec, np.corrcoef(s[ok], rem[ok])[0, 1],
                         np.mean(np.sign(s[ok]) * rem[ok]) * 1e4,
                         np.mean((np.sign(s[ok]) == np.sign(rem[ok]))[rem[ok] != 0])))
    D = pd.DataFrame(rows, columns=["day", "delay_s", "ic", "edge_bp", "hit"])
    print(f"\n=== horizon {h}m: signal quality vs execution delay after bar close ===")
    print(D.groupby("delay_s")[["ic", "edge_bp", "hit"]].mean().round(4).to_string())
