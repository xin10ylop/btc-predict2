"""Download Binance public bulk data (data.binance.vision) used for research/training.

Usage: python scripts/download.py --start 2021-01 --end 2026-10
"""
import argparse, os, sys, io, zipfile, concurrent.futures as cf
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from btcpred import paths  # noqa: E402

BASE = "https://data.binance.vision/data"
DATASETS = {
    # name: (path template, filename template)
    "spot_1m":    ("spot/monthly/klines/BTCUSDT/1m", "BTCUSDT-1m-{m}.zip"),
    "perp_1m":    ("futures/um/monthly/klines/BTCUSDT/1m", "BTCUSDT-1m-{m}.zip"),
    "premium_1m": ("futures/um/monthly/premiumIndexKlines/BTCUSDT/1m", "BTCUSDT-1m-{m}.zip"),
    "funding":    ("futures/um/monthly/fundingRate/BTCUSDT", "BTCUSDT-fundingRate-{m}.zip"),
    "eth_spot_1m": ("spot/monthly/klines/ETHUSDT/1m", "ETHUSDT-1m-{m}.zip"),
    "index_1m":   ("futures/um/monthly/indexPriceKlines/BTCUSDT/1m", "BTCUSDT-1m-{m}.zip"),
}


def months(start, end):
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m == 13:
            y, m = y + 1, 1


def fetch(url, out):
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out, "cached"
    r = requests.get(url, timeout=120)
    if r.status_code != 200:
        return out, f"HTTP {r.status_code}"
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = z.namelist()[0]
        data = z.read(name)
    tmp = out + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, out)
    return out, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01")
    ap.add_argument("--end", default="2026-10")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--out", default=str(paths.RAW))
    a = ap.parse_args()
    jobs = []
    for ds in a.datasets.split(","):
        path, fn = DATASETS[ds]
        d = os.path.join(a.out, ds)
        os.makedirs(d, exist_ok=True)
        for m in months(a.start, a.end):
            f = fn.format(m=m)
            jobs.append((f"{BASE}/{path}/{f}", os.path.join(d, f.replace(".zip", ".csv"))))
    with cf.ThreadPoolExecutor(8) as ex:
        for out, st in ex.map(lambda j: fetch(*j), jobs):
            if st != "cached":
                print(st, out, flush=True)
    print("done", len(jobs))


if __name__ == "__main__":
    sys.exit(main())
