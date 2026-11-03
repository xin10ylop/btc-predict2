"""Download Binance spot 1s klines month by month and reduce them to the price at
+d seconds after every minute close (d in OFFSETS). Raw files are discarded.
Output: data/sec/YYYY-MM.parquet indexed by the 1m bar open time t; column px_d =
spot price at (t + 60s + d) i.e. d seconds after bar t closed (px_0 == bar close)."""
import io, os, sys, zipfile, concurrent.futures as cf
import numpy as np, pandas as pd, requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from btcpred import paths  # noqa: E402
OFFSETS = [0, 1, 2, 3, 5, 10, 20, 30]
URL = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1s/BTCUSDT-1s-{m}.zip"
OUT = str(paths.SEC)

def months(a, b):
    y, m = map(int, a.split("-")); ey, em = map(int, b.split("-"))
    while (y, m) <= (ey, em):
        yield f"{y:04d}-{m:02d}"; m += 1
        if m == 13: y, m = y + 1, 1

def job(m):
    out = f"{OUT}/{m}.parquet"
    if os.path.exists(out): return m, "cached"
    r = requests.get(URL.format(m=m), timeout=600)
    if r.status_code != 200: return m, f"HTTP {r.status_code}"
    z = zipfile.ZipFile(io.BytesIO(r.content))
    df = pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=[0, 4], names=["t", "c"])
    t = df["t"].astype("int64").values
    t = np.where(t > 10**14, t // 1000, t) // 1000  # seconds
    c = pd.Series(df["c"].values, index=t)
    c = c[~c.index.duplicated()]
    full = np.arange(t.min(), t.max() + 1)
    c = c.reindex(full).ffill()  # 1s bars with no trades are omitted -> carry last price
    # 1s bar opening at second s closes at s+1. price at time T is close of bar opening at T-1.
    mins = np.unique((full // 60) * 60)
    res = {}
    for d in OFFSETS:
        res[f"px_{d}"] = c.reindex(mins + 60 + d - 1).values
    o = pd.DataFrame(res, index=pd.to_datetime(mins, unit="s", utc=True)).astype("float64")
    o.to_parquet(out)
    return m, "ok"

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    ms = list(months(sys.argv[1], sys.argv[2]))
    with cf.ThreadPoolExecutor(3) as ex:
        for m, st in ex.map(job, ms):
            print(m, st, flush=True)
