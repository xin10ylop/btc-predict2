"""Live BTC predictor: next 1..5 minutes, refreshed every minute.

    python scripts/predict_live.py            # loop forever, one forecast per minute
    python scripts/predict_live.py --once     # single forecast

Every forecast is appended to logs/live_predictions.csv and scored automatically once the
outcome is known, so you can watch the live hit-rate yourself.
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from btcpred.features import make_features            # noqa: E402
from btcpred.model import BTCPredictor, HORIZONS       # noqa: E402
from btcpred import live                                # noqa: E402

PLAIN = {
    "basis_dev_60": "perp-vs-spot gap (perp leads)", "basis_chg_1": "perp-vs-spot gap change",
    "basis_dev_1440": "perp-vs-spot gap vs day", "basis_bp": "perp basis level",
    "premium_dev_60": "perp premium index", "premium_bp": "premium level",
    "vwap_dev_30": "stretch from 30m VWAP", "vwap_dev_240": "stretch from 4h VWAP",
    "ret_30": "30m move (reverts)", "ret_60": "60m move", "ret_15": "15m move", "ret_10": "10m move",
    "dist_hi_60": "distance below 1h high", "dist_lo_60": "distance above 1h low",
    "breakout_60": "1h breakout (tends to fail)", "sweep_60": "1h liquidity sweep",
    "impact_expl_15": "flow-driven move 15m (reverts)", "impact_resid_15": "flow-free move 15m",
    "p_flow_15": "perp aggressive flow 15m", "s_flow_15": "spot aggressive flow 15m",
    "eth_rel_1": "ETH vs BTC last minute", "eth_rel_5": "ETH vs BTC 5m", "minute": "minute-of-hour clock",
}


def explain(pred, X_last, h, k=4):
    m = pred.dir_models[h][len(pred.dir_models[h]) // 2]
    c = m.predict(X_last[pred.features].values, pred_contrib=True)[0][:-1]
    order = np.argsort(-np.abs(c))[:k]
    out = []
    for i in order:
        f = pred.features[i]
        out.append(f"{'+' if c[i] > 0 else '-'} {PLAIN.get(f, f)} ({X_last[f].iloc[0]:+.2f})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/live")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--log", default="logs/live_predictions.csv")
    a = ap.parse_args()
    pred = BTCPredictor.load(a.model)
    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    print(f"model trained until {pred.meta.get('trained_until')} | act within ~{pred.delay}s of the candle close")
    cache = None
    while True:
        if not a.once:  # wake 0.4s after the minute boundary
            now = time.time()
            time.sleep(60 - (now % 60) + 0.4)
        t0 = time.time()
        try:
            if cache is None:
                cache = live.fetch_panel(8000)
            else:
                new = live.fetch_panel(5)
                cache = pd.concat([cache, new])
                cache = cache[~cache.index.duplicated(keep="last")].sort_index().tail(8000)
                full = pd.date_range(cache.index[0], cache.index[-1], freq="1min", tz="UTC")
                cache = cache.reindex(full)
            X, sig = make_features(cache)
            px_now = live.spot_price_now()
        except Exception as e:  # network hiccup: skip this minute
            print("data error:", e, flush=True)
            if a.once:
                raise
            continue
        bar = X.index[-1]
        last = X.iloc[[-1]]
        f = pred.describe(last, px_now, float(sig.iloc[-1]))
        lag = time.time() - (bar + pd.Timedelta(minutes=1)).timestamp()
        print(f"\n{pd.Timestamp.utcnow():%Y-%m-%d %H:%M:%S} UTC | BTC {px_now:,.2f} | last candle {bar:%H:%M} "
              f"| data age {lag:.1f}s")
        print("  horizon  P(up)   confidence      median move   90% range")
        for h in HORIZONS:
            r = f[h]
            c = r["confidence_pct"]
            tag = "top 10%" if c >= 90 else ("top 25%" if c >= 75 else "      ")
            print(f"   {h}m     {r['p_up']*100:5.1f}%  {c:5.1f} pct {tag}  {r['median_move_bp']:+6.2f} bp   "
                  f"{r['price_q05']:,.0f} – {r['price_q95']:,.0f}")
        for h in (1, 5):
            print(f"  why {h}m: " + "; ".join(explain(pred, last, h)))
        if last["clock_class"].iloc[0] == 4 and abs(last["vwap_dev_30"].iloc[0]) > 1.0:
            side = "DOWN" if last["vwap_dev_30"].iloc[0] > 0 else "UP"
            print(f"  ! funding-settlement window and price is {last['vwap_dev_30'].iloc[0]:+.1f} sigma from its 30m VWAP:"
                  f" since 2023 such gaps reverted 3-4x harder than usual (leans {side}; informational, not in the model)")
        row = dict(bar=bar.isoformat(), made_at=pd.Timestamp.utcnow().isoformat(), price=px_now,
                   **{f"p_up_{h}": f[h]["p_up"] for h in HORIZONS})
        pd.DataFrame([row]).to_csv(a.log, mode="a", header=not os.path.exists(a.log), index=False)
        # live scorecard: compare logged forecasts with realised closes from the cache
        L = pd.read_csv(a.log)
        L["bar"] = pd.to_datetime(L["bar"], utc=True)
        closes = cache["s_close"]
        hits = {}
        for h in (1, 5):
            tgt = closes.reindex(L["bar"] + pd.Timedelta(minutes=h)).values
            ok = np.isfinite(tgt) & (tgt != L["price"].values)  # unchanged price = no call
            if ok.sum() >= 10:
                up = tgt[ok] > L["price"].values[ok]
                hits[h] = (np.mean((L[f"p_up_{h}"].values[ok] > 0.5) == up), int(ok.sum()))
        if hits:
            print("  live scorecard: " + " | ".join(f"{h}m hit {v[0]*100:.1f}% (n={v[1]})" for h, v in hits.items()))
        if a.once:
            break


if __name__ == "__main__":
    main()
