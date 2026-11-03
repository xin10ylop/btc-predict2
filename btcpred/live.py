"""Live data access + live prediction loop.

Endpoints (public market data, no key needed). Several hosts are tried in order because
api.binance.com / fapi.binance.com are geo-restricted in some countries.
"""
import time
import numpy as np
import pandas as pd
import requests

SPOT_HOSTS = ["https://data-api.binance.vision", "https://api.binance.com", "https://www.binance.com"]
PERP_HOSTS = ["https://fapi.binance.com", "https://www.binance.com"]
KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
         "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
NUM = ["open", "high", "low", "close", "volume", "quote_volume", "count",
       "taker_buy_volume", "taker_buy_quote_volume"]
_session = requests.Session()
_good_host = {}


def _get(kind, path, params):
    hosts = PERP_HOSTS if kind == "perp" else SPOT_HOSTS
    if kind in _good_host:
        hosts = [_good_host[kind]] + [h for h in hosts if h != _good_host[kind]]
    last = None
    for h in hosts:
        try:
            r = _session.get(h + path, params=params, timeout=10)
            if r.status_code == 200:
                _good_host[kind] = h
                return r.json()
            last = f"{h}: HTTP {r.status_code}"
        except requests.RequestException as e:
            last = f"{h}: {e}"
    raise RuntimeError(f"all hosts failed for {path}: {last}")


def klines(kind, path, symbol, n, key="symbol"):
    """Last n CLOSED 1m klines as a DataFrame indexed by open time (UTC)."""
    rows, end = [], None
    while len(rows) < n + 1:
        p = {key: symbol, "interval": "1m", "limit": min(1000, n + 2)}
        if end is not None:
            p["endTime"] = end
        batch = _get(kind, path, p)
        if not batch:
            break
        rows = batch + rows
        end = int(batch[0][0]) - 1
        if len(batch) < p["limit"]:
            break
    df = pd.DataFrame(rows, columns=KCOLS)
    df = df.drop_duplicates("open_time").sort_values("open_time")
    now_ms = int(time.time() * 1000)
    df = df[df["close_time"].astype("int64") < now_ms]  # closed bars only
    df.index = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df.index.name = "ts"
    return df[NUM].astype("float64").tail(n)


def fetch_panel(n=8000):
    """Same column layout as btcpred.data.build_panel."""
    spot = klines("spot", "/api/v3/klines", "BTCUSDT", n).add_prefix("s_")
    eth = klines("spot", "/api/v3/klines", "ETHUSDT", n)[["close", "volume", "quote_volume",
                                                          "taker_buy_quote_volume"]].add_prefix("e_")
    perp = klines("perp", "/fapi/v1/klines", "BTCUSDT", n).add_prefix("p_")
    prem = klines("perp", "/fapi/v1/premiumIndexKlines", "BTCUSDT", n)[["open", "high", "low", "close"]].add_prefix("pi_")
    idx = pd.date_range(spot.index[0], spot.index[-1], freq="1min", tz="UTC")
    P = pd.concat([x.reindex(idx) for x in (spot, perp, prem, eth)], axis=1)
    P.index.name = "ts"
    return P


def spot_price_now():
    return float(_get("spot", "/api/v3/ticker/price", {"symbol": "BTCUSDT"})["price"])


def seconds_prices(close_ts, n=20):
    """Spot price at close_ts + d seconds for d = 1..n (from 1-second klines).
    Returns {d: price}. The 1s candle opening at close_ts + (d-1) s closes at close_ts + d s."""
    start = int(pd.Timestamp(close_ts).timestamp() * 1000)
    rows = _get("spot", "/api/v3/klines", {"symbol": "BTCUSDT", "interval": "1s", "startTime": start, "limit": n})
    px = {}
    for r in rows:
        d = (int(r[0]) - start) // 1000 + 1
        if 1 <= d <= n:
            px[d] = float(r[4])
    # seconds without trades have no candle: carry the last price forward
    last = None
    for d in range(1, n + 1):
        if d in px:
            last = px[d]
        elif last is not None:
            px[d] = last
    return px
