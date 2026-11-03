"""Loading and aligning Binance 1-minute data.

Everything is indexed by the bar OPEN time (UTC, minute resolution). A bar with
open time t is fully known at t+1min (its close). Features computed at row t use
only rows <= t; targets at row t use rows > t.
"""
import glob, os
import numpy as np
import pandas as pd

from . import paths

KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
         "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def _read_kline_csv(path):
    with open(path) as f:
        first = f.readline()
    header = 0 if first.startswith("open_time") else None
    df = pd.read_csv(path, header=header, names=None if header == 0 else KCOLS)
    df.columns = KCOLS
    t = df["open_time"].astype("int64")
    # Binance switched spot timestamps to microseconds on 2025-01-01
    t = np.where(t > 10**14, t // 1000, t)
    df.index = pd.to_datetime(t, unit="ms", utc=True)
    df.index.name = "ts"
    return df[["open", "high", "low", "close", "volume", "quote_volume", "count",
               "taker_buy_volume", "taker_buy_quote_volume"]].astype("float64")


def load_klines(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    df = pd.concat([_read_kline_csv(f) for f in files])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def load_funding(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    out = []
    for f in files:
        d = pd.read_csv(f)
        d.index = pd.DatetimeIndex(pd.to_datetime(d["calc_time"].astype("int64"), unit="ms", utc=True)).floor("min")
        out.append(d[["last_funding_rate"]])
    d = pd.concat(out).sort_index()
    return d[~d.index.duplicated(keep="last")]


def build_panel(raw=paths.RAW, out=paths.PANEL):
    spot = load_klines(f"{raw}/spot_1m").add_prefix("s_")
    perp = load_klines(f"{raw}/perp_1m").add_prefix("p_")
    prem = load_klines(f"{raw}/premium_1m")[["open", "high", "low", "close"]].add_prefix("pi_")
    parts = [spot, perp, prem]
    if os.path.isdir(f"{raw}/index_1m"):
        parts.append(load_klines(f"{raw}/index_1m")[["close"]].add_prefix("ix_"))
    if os.path.isdir(f"{raw}/eth_spot_1m"):
        eth = load_klines(f"{raw}/eth_spot_1m")[["close", "volume", "quote_volume",
                                                 "taker_buy_quote_volume"]].add_prefix("e_")
        parts.append(eth)
    idx = pd.date_range(spot.index[0], spot.index[-1], freq="1min", tz="UTC")
    panel = pd.concat([p.reindex(idx) for p in parts], axis=1)
    panel.index.name = "ts"
    fund = load_funding(f"{raw}/funding")
    # funding rate realized at settlement; known from that minute on
    panel["fund_rate"] = fund["last_funding_rate"].reindex(idx, method=None)
    panel["fund_rate"] = panel["fund_rate"].ffill()
    panel["s_missing"] = panel["s_close"].isna().astype("int8")
    panel["p_missing"] = panel["p_close"].isna().astype("int8")
    panel = panel.astype({c: "float32" for c in panel.columns if c not in ("s_missing", "p_missing")})
    panel.to_parquet(out)
    return panel


def load_panel(columns=None, before=None):
    """The aligned 1-minute panel; `before` (e.g. HOLDOUT_START) cuts it."""
    P = pd.read_parquet(paths.PANEL, columns=columns)
    return P[P.index < before] if before is not None else P


def load_sec_offsets():
    """Spot price d seconds after every minute close (columns px_0, px_1, ...), all months."""
    files = sorted(glob.glob(str(paths.SEC / "*.parquet")))
    S = pd.concat([pd.read_parquet(f) for f in files])
    return S[~S.index.duplicated()].sort_index()


def load_dev(columns=None, targets=None):
    """Research cache: features X, targets T and the feature->family map (development period)."""
    X = pd.read_parquet(paths.X_DEV, columns=columns)
    T = pd.read_parquet(paths.T_DEV, columns=targets)
    fam = pd.read_csv(paths.FAMILIES, index_col=0).iloc[:, 0].to_dict()
    return X, T, fam


if __name__ == "__main__":
    p = build_panel()
    print(p.shape, p.index[0], p.index[-1])
    print(p[["s_missing", "p_missing"]].sum())
    print(p.isna().mean().round(4).to_string())
