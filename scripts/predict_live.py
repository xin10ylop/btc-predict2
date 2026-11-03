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
from btcpred import paths                               # noqa: E402
from btcpred.model import BTCPredictor, h_label, is_sec  # noqa: E402
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
    "eth_rel_1": "ETH vs BTC last minute", "eth_rel_5": "ETH vs BTC 5m", "eth_rel_15": "ETH vs BTC 15m",
    "minute": "minute-of-hour clock", "min_mod15": "position in quarter-hour", "hour": "hour of day (UTC)",
    "dow": "day of week", "to_funding": "minutes to funding settlement",
    "basis_fresh": "fresh perp-spot gap (opened late in the bar)", "basis_vwap_dev60": "perp-spot gap at bar VWAP",
    "p_close_vs_vwap": "perp close vs its bar VWAP", "s_close_vs_vwap": "spot close vs its bar VWAP",
    "ret_1": "last-minute move", "ret_2": "2m move", "ret_3": "3m move", "ret_5": "5m move", "ret_240": "4h move",
    "zlag_1": "previous minute's move", "zlag_2": "move 2 minutes ago",
    "dist_hi_15": "distance below 15m high", "dist_lo_15": "distance above 15m low",
    "dist_hi_240": "distance below 4h high", "dist_lo_240": "distance above 4h low",
    "dist_hi_1440": "distance below 24h high", "dist_lo_1440": "distance above 24h low",
    "clv": "where the candle closed in its range", "upper_wick": "upper wick", "lower_wick": "lower wick",
    "vol_ratio_10_60": "volatility rising/falling", "vol_ratio_60_1440": "volatility vs day", "sig_bp": "volatility level",
    "range_last": "last candle range", "range_5": "5m range",
    "rg_leadlag_1d": "perp-lead strength, last 24h", "rg_leadlag_3d": "perp-lead strength, last 3d",
    "rg_fresh_1d": "gap-freshness strength, last 24h", "rg_micromom_1d": "minute momentum regime, 24h",
    "rg_micromom_3d": "minute momentum regime, 3d", "rg_clv_1d": "close-location regime, 24h",
    "rg_meanrev_3d": "mean-reversion strength, 3d", "rg_vol_1d_vs_5d": "volatility vs past 5 days",
    "rg_liq_1d_vs_5d": "volume vs past 5 days",
}
HISTORY = 20000   # minutes of history kept live: features need ~8000, the 7-day self-check needs the rest


def regime_line(last):
    v, q, ll = (float(last[c].iloc[0]) for c in ("rg_vol_1d_vs_5d", "rg_liq_1d_vs_5d", "rg_leadlag_1d"))
    vol = "calmer than this week" if v < -0.25 else ("wilder than this week" if v > 0.25 else "normal for this week")
    liq = "thin" if q < -0.25 else ("heavy" if q > 0.25 else "normal")
    lead = "strong" if ll > 0.10 else ("normal" if ll > 0.04 else "weak/absent")
    return f"regime: volatility {vol} ({v:+.2f}) | volume {liq} | perp-lead last 24h {lead} ({ll:+.3f})"


def show_track(pred):
    if not pred.track:
        print("  self-check: fewer than ~1.5 days since the model's training cutoff, using the static calibration")
        return
    parts = [f"{h}m hit {t['hit']*100:.1f}% skill {t['skill_bp']:+.0f}bp temp {t['temp']:.2f}{'*' if t['online'] else ''}"
             for h, t in sorted(pred.track.items())]
    print(f"  self-check, last {pred.track[min(pred.track)]['days']} days: " + " | ".join(parts))
    print("  (* = temperature refitted on this self-check window; 1m always, other horizons only when their edge broke)")
    bad = [h for h, t in pred.track.items() if t["skill_bp"] < 0]
    if bad:
        print(f"  ! edge NOT confirmed in the last 7 days for {', '.join(f'{h}m' for h in sorted(bad))} (regime change?):"
              " those probabilities are shrunk toward 50%")


def explain(pred, X_last, h, k=4):
    m = pred.dir_models[h][len(pred.dir_models[h]) // 2]
    fh = pred.feats(h)
    c = m.predict(X_last[fh].values, pred_contrib=True)[0][:-1]
    order = np.argsort(-np.abs(c))[:k]
    out = []
    for i in order:
        f = fh[i]
        out.append(f"{'+' if c[i] > 0 else '-'} {PLAIN.get(f, f)} ({X_last[f].iloc[0]:+.2f})")
    return out


def score_seconds(pending, path, delay=2):
    """Outcomes of 10s/15s forecasts made at earlier candle closes: price at close+delay+H vs
    close+delay, from 1-second candles. Unchanged price -> no call (NaN)."""
    for bar in sorted(pending):
        close = bar + pd.Timedelta(minutes=1)
        if time.time() < close.timestamp() + 20:
            continue
        try:
            px = live.seconds_prices(close, 20)
        except Exception:
            continue
        row = {"bar": bar.isoformat()}
        for h, p in pending[bar].items():
            H = int(h[1:])
            a_, b_ = px.get(delay), px.get(delay + H)
            row[f"p_up_{h}"] = p
            row[f"y_{h}"] = np.nan if (a_ is None or b_ is None or a_ == b_) else float(b_ > a_)
        pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        del pending[bar]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.LIVE_MODEL))
    ap.add_argument("--once", action="store_true", help="one forecast at the next candle close")
    ap.add_argument("--now", action="store_true", help="with --once: forecast immediately (may be stale)")
    ap.add_argument("--log", default=str(paths.LIVE_LOG))
    a = ap.parse_args()
    pred = BTCPredictor.load(a.model)
    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    print(f"model trained until {pred.meta.get('trained_until')} | act within ~{pred.delay}s of the candle close")
    print(f"loading history ({HISTORY} minutes)...", flush=True)
    cache = live.fetch_panel(HISTORY)  # warm-up before the first wait, so the first forecast is on time
    X, _ = make_features(cache)
    pred.online_calibrate(X, cache["s_close"])   # regime adaptation: self-check on the last 7 days
    show_track(pred)
    last_recal = time.time()
    first = True
    pending = {}                                            # bar -> seconds-horizon forecasts awaiting outcome
    sec_log = os.path.join(os.path.dirname(a.log), "live_seconds.csv")
    while True:
        if not (a.once and a.now):  # wake 0.4s after the minute boundary
            now = time.time()
            time.sleep(60 - (now % 60) + 0.4)
        t0 = time.time()
        try:
            if first and a.once and a.now:
                pass  # cache is fresh enough
            else:
                new = live.fetch_panel(5)
                cache = pd.concat([cache, new])
                cache = cache[~cache.index.duplicated(keep="last")].sort_index().tail(HISTORY)
                full = pd.date_range(cache.index[0], cache.index[-1], freq="1min", tz="UTC")
                cache = cache.reindex(full)
            X, sig = make_features(cache)
            px_now = live.spot_price_now()
        except Exception as e:  # network hiccup: skip this minute
            print("data error:", e, flush=True)
            if a.once:
                raise
            continue
        first = False
        if time.time() - last_recal > 3600:  # hourly: re-score the last 7 days, refit temperatures
            pred.online_calibrate(X, cache["s_close"])
            last_recal = time.time()
            show_track(pred)
        bar = X.index[-1]
        last = X.iloc[[-1]]
        f = pred.describe(last, px_now, float(sig.iloc[-1]))
        lag = time.time() - (bar + pd.Timedelta(minutes=1)).timestamp()
        print(f"\n{pd.Timestamp.now("UTC"):%Y-%m-%d %H:%M:%S} UTC | BTC {px_now:,.2f} | last candle {bar:%H:%M} "
              f"| data age {lag:.1f}s")
        if lag > 10:
            print(f"  (data is {lag:.0f}s old; the model assumes you act ~{pred.delay}s after the close, so the 1m edge is mostly gone)")
        print("  horizon  P(up)   confidence      median move   90% range")
        for h in pred.horizons:
            r = f[h]
            c = r["confidence_pct"]
            tag = "top 10%" if c >= 90 else ("top 25%" if c >= 75 else "      ")
            flat = f"   no change {r['p_flat']*100:4.1f}%" if is_sec(h) and r["p_flat"] == r["p_flat"] else ""
            print(f"   {h_label(h):>3}     {r['p_up']*100:5.1f}%  {c:5.1f} pct {tag}  {r['median_move_bp']:+6.2f} bp   "
                  f"{r['price_q05']:,.0f} – {r['price_q95']:,.0f}{flat}")
        if any(is_sec(h) for h in pred.horizons):
            print("  (10s/15s: counted from ~2s after the candle close; P(up) is given that the price moves)")
        for h in [h for h in ("s10", 1, 5) if h in pred.horizons]:
            print(f"  why {h_label(h)}: " + "; ".join(explain(pred, last, h)))
        print("  " + regime_line(last))
        if last["clock_class"].iloc[0] == 4 and abs(last["vwap_dev_30"].iloc[0]) > 1.0:
            side = "DOWN" if last["vwap_dev_30"].iloc[0] > 0 else "UP"
            print(f"  ! funding-settlement window and price is {last['vwap_dev_30'].iloc[0]:+.1f} sigma from its 30m VWAP:"
                  f" since 2023 such gaps reverted 3-4x harder than usual (leans {side}; informational, not in the model)")
        row = dict(bar=bar.isoformat(), made_at=pd.Timestamp.now("UTC").isoformat(), price=px_now,
                   **{f"p_up_{h}": f[h]["p_up"] for h in pred.horizons})
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
        # seconds horizons: score the previous forecasts against 1-second candles
        if any(is_sec(h) for h in pred.horizons):
            score_seconds(pending, sec_log)
            pending[bar] = {h: f[h]["p_up"] for h in pred.horizons if is_sec(h)}
            if os.path.exists(sec_log):
                S = pd.read_csv(sec_log)
                for h in [h for h in pred.horizons if is_sec(h)]:
                    m = S[f"y_{h}"].notna()
                    if m.sum() >= 10:
                        hits[h] = (np.mean((S.loc[m, f"p_up_{h}"] > 0.5) == (S.loc[m, f"y_{h}"] > 0)), int(m.sum()))
        if hits:
            print("  live scorecard: " + " | ".join(f"{h_label(h)} hit {v[0]*100:.1f}% (n={v[1]})" for h, v in hits.items()))
        if a.once:
            break


if __name__ == "__main__":
    main()
