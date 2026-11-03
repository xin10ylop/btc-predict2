"""Feature engineering. Shared by research, training and the live predictor.

Convention: row t = 1-minute bar that OPENED at t (closes at t+1min). Every
feature at row t uses only bars <= t, i.e. it is known at the close of bar t.

Every price feature is expressed in units of the current volatility estimate
(sigma), so the model sees the same scale in a $20k market and a $120k market,
in a quiet 2023 and a violent 2021. Volume features are expressed relative to
their own recent history for the same reason (Binance fee regimes changed raw
volume/trade counts by multiples between 2021 and 2026).
"""
import numpy as np
import pandas as pd

EPS = 1e-12

# feature families, filled in by make_features (name -> family)
FAMILY = {}


def _add(F, name, values, family):
    F[name] = np.asarray(values, dtype="float32")
    FAMILY[name] = family


def ewm(x, span):
    return x.ewm(span=span, adjust=False, min_periods=max(5, span // 4)).mean()


def sigma_1m(logc, span=60):
    """Per-minute volatility known at t (EWMA of squared returns, floor for dead bars)."""
    r = logc.diff()
    v = np.sqrt(ewm(r * r, span))
    return v.clip(lower=1e-5), r


def make_features(P, families=None):
    """P: panel with s_*, p_*, pi_*, (e_*) columns indexed by minute."""
    F = {}
    idx = P.index
    s_c = np.log(P["s_close"].astype("float64"))
    s_h = np.log(P["s_high"].astype("float64"))
    s_l = np.log(P["s_low"].astype("float64"))
    s_o = np.log(P["s_open"].astype("float64"))
    sig, r = sigma_1m(s_c, 60)
    sig_slow, _ = sigma_1m(s_c, 1440)
    z1 = (r / sig.shift(1)).clip(-10, 10)

    # ---------------- A. returns / momentum at several scales -------------
    for k in (1, 2, 3, 5, 10, 15, 30, 60, 240):
        _add(F, f"ret_{k}", (s_c - s_c.shift(k)) / (sig * np.sqrt(k)), "returns")
    for lag in (1, 2):  # individual earlier bars
        _add(F, f"zlag_{lag}", z1.shift(lag), "returns")

    # ---------------- B. volatility state ---------------------------------
    sig10, _ = sigma_1m(s_c, 10)
    _add(F, "vol_ratio_10_60", np.log(sig10 / sig), "vol")
    _add(F, "vol_ratio_60_1440", np.log(sig / sig_slow), "vol")
    _add(F, "sig_bp", sig * 1e4, "vol")
    park = (s_h - s_l) / sig
    _add(F, "range_last", park, "vol")
    _add(F, "range_5", (s_h.rolling(5).max() - s_l.rolling(5).min()) / (sig * np.sqrt(5)), "vol")

    # ---------------- C. bar shape / range position -----------------------
    hl = (s_h - s_l).clip(lower=EPS)
    _add(F, "clv", ((s_c - s_l) - (s_h - s_c)) / hl, "shape")  # close location value
    _add(F, "upper_wick", (s_h - np.maximum(s_c, s_o)) / sig, "shape")
    _add(F, "lower_wick", (np.minimum(s_c, s_o) - s_l) / sig, "shape")
    for n in (15, 60, 240, 1440):
        hi = s_h.rolling(n).max()
        lo = s_l.rolling(n).min()
        _add(F, f"dist_hi_{n}", (hi - s_c) / (sig * np.sqrt(n)), "range")
        _add(F, f"dist_lo_{n}", (s_c - lo) / (sig * np.sqrt(n)), "range")

    # VWAP deviation (rolling) -- where the average participant's cost basis is
    qv = P["s_quote_volume"].astype("float64")
    vol = P["s_volume"].astype("float64")
    for n in (30, 240):
        vwap = np.log(qv.rolling(n).sum() / vol.rolling(n).sum().clip(lower=EPS))
        _add(F, f"vwap_dev_{n}", (s_c - vwap) / (sig * np.sqrt(n)), "range")

    # ---------------- D. order flow (spot and perp) -----------------------
    s_tb = P["s_taker_buy_quote_volume"].astype("float64")
    s_flow = 2 * s_tb - qv  # signed aggressive dollar flow
    s_qv_avg = ewm(qv, 1440).clip(lower=1.0)
    p_qv = P["p_quote_volume"].astype("float64")
    p_tb = P["p_taker_buy_quote_volume"].astype("float64")
    p_flow = 2 * p_tb - p_qv
    p_qv_avg = ewm(p_qv, 1440).clip(lower=1.0)
    for name, flow, tot, avg in (("s", s_flow, qv, s_qv_avg), ("p", p_flow, p_qv, p_qv_avg)):
        _add(F, f"{name}_imb_1", flow / tot.clip(lower=1.0), "flow")
        for n in (5, 15, 60):
            _add(F, f"{name}_imb_{n}", flow.rolling(n).sum() / tot.rolling(n).sum().clip(lower=1.0), "flow")
        _add(F, f"{name}_flow_1", flow / avg, "flow")
        _add(F, f"{name}_flow_15", flow.rolling(15).sum() / (avg * 15), "flow")
        _add(F, f"{name}_vsurp_1", np.log((tot + 1) / (ewm(tot, 60) + 1)), "flow")
        _add(F, f"{name}_vsurp_60", np.log((ewm(tot, 60) + 1) / avg), "flow")
    s_cnt = P["s_count"].astype("float64").clip(lower=1)
    tsize = qv / s_cnt
    _add(F, "s_tradesize_surp", np.log((tsize + 1) / (ewm(tsize, 1440) + 1)), "flow")
    p_cnt = P["p_count"].astype("float64").clip(lower=1)
    ptsize = p_qv / p_cnt
    _add(F, "p_tradesize_surp", np.log((ptsize + 1) / (ewm(ptsize, 1440) + 1)), "flow")
    _add(F, "perp_spot_vol_share", np.log((p_qv + 1) / (qv + 1)) - np.log((ewm(p_qv, 1440) + 1) / (ewm(qv, 1440) + 1)), "flow")

    # ---------------- E. perp vs spot (basis, lead-lag) -------------------
    p_c = np.log(P["p_close"].astype("float64"))
    basis = (p_c - s_c)
    _add(F, "basis_dev_60", (basis - ewm(basis, 60)) / sig, "basis")
    _add(F, "basis_dev_1440", (basis - ewm(basis, 1440)) / sig, "basis")
    _add(F, "basis_chg_1", basis.diff() / sig, "basis")
    _add(F, "basis_bp", basis * 1e4, "basis")
    pi = P["pi_close"].astype("float64")
    _add(F, "premium_dev_60", (pi - ewm(pi, 60)) / sig, "basis")
    _add(F, "premium_bp", pi * 1e4, "basis")

    # ---------------- F. calendar / clock ---------------------------------
    m = idx.minute.values
    _add(F, "minute", m, "clock")
    _add(F, "min_mod15", m % 15, "clock")
    _add(F, "hour", idx.hour.values, "clock")
    _add(F, "dow", idx.dayofweek.values, "clock")
    # minutes until next 8h funding settlement (00,08,16 UTC)
    mins_day = idx.hour.values * 60 + m
    _add(F, "to_funding", (480 - (mins_day % 480)) % 480, "clock")

    # clock-conditioned reversion: deviations from VWAP revert ~2x harder before quarter-hours
    # and ~3-4x harder within +-1 min of the 8h funding settlement (2023+). Settlement rows are
    # too rare (0.6%) for regularised trees to isolate, so the interaction is made explicit.
    tf = F["to_funding"]
    settle = (tf <= 2).astype("float32")                   # bars 07:58, 07:59, 08:00 (and 00/16h)
    preq = ((m % 15) == 14).astype("float32")
    w = 1.0 + preq + 3.0 * settle
    _add(F, "clock_class", np.select([settle > 0, preq > 0, (m % 15) == 0, (m % 5) == 4], [4, 3, 2, 1], 0), "clockrev")
    _add(F, "vwap30_x_clock", F["vwap_dev_30"] * w, "clockrev")
    _add(F, "ret15_x_clock", F["ret_15"] * w, "clockrev")

    # ---------------- G. cross-asset (ETH) --------------------------------
    if "e_close" in P:
        e_c = np.log(P["e_close"].astype("float64"))
        esig, er = sigma_1m(e_c, 60)
        for k in (1, 5, 15):
            e_ret = (e_c - e_c.shift(k)) / (esig * np.sqrt(k))
            _add(F, f"eth_rel_{k}", e_ret - F[f"ret_{k}"], "eth")


    # ---------------- L. intra-bar structure recovered from kline volume fields ---
    # bar VWAP = quote_volume / volume. Close vs VWAP tells how the minute ENDED;
    # close-basis minus VWAP-basis tells whether the perp/spot dislocation is FRESH
    # (appeared late in the minute, spot has not caught up yet) or stale.
    s_bvwap = np.log(qv / vol.where(vol > 0))
    p_vol = P["p_volume"].astype("float64")
    p_bvwap = np.log(p_qv / p_vol.where(p_vol > 0))
    _add(F, "s_close_vs_vwap", (s_c - s_bvwap) / sig, "intrabar")
    _add(F, "p_close_vs_vwap", (p_c - p_bvwap) / sig, "intrabar")
    _add(F, "basis_fresh", (basis - (p_bvwap - s_bvwap)) / sig, "intrabar")
    vbasis = p_bvwap - s_bvwap
    _add(F, "basis_vwap_dev60", (vbasis - ewm(vbasis, 60)) / sig, "intrabar")

    # ---------------- H. transient vs permanent impact (Hasbrouck / propagator) ---
    # Rolling regression of spot return on combined signed flow. Finding: the
    # flow-explained part of the last ~15 min reverts; the flow-free part does not.
    flow_n = (s_flow + p_flow) / (s_qv_avg + p_qv_avg)
    rr = r.fillna(0.0)
    fx = flow_n.fillna(0.0)
    W = 1440
    cov = (rr * fx).rolling(W, min_periods=300).mean() - rr.rolling(W, min_periods=300).mean() * fx.rolling(W, min_periods=300).mean()
    var = (fx * fx).rolling(W, min_periods=300).mean() - fx.rolling(W, min_periods=300).mean() ** 2
    lam = (cov / var.clip(lower=1e-12)).shift(1)  # Kyle-lambda, known before bar t
    expl = lam * fx
    resid = rr - expl
    for n in (1, 5, 15, 60):
        _add(F, f"impact_expl_{n}", expl.rolling(n).sum() / (sig * np.sqrt(n)), "impact")
        _add(F, f"impact_resid_{n}", resid.rolling(n).sum() / (sig * np.sqrt(n)), "impact")
    _add(F, "kyle_lambda_z", (lam / sig), "impact")

    # ---------------- I. liquidity sweeps / failed breakouts --------------
    for n in (60, 240, 1440):
        prev_hi = s_h.shift(1).rolling(n).max()
        prev_lo = s_l.shift(1).rolling(n).min()
        # poked above the prior n-min high but closed back below it (and vice versa)
        sweep_hi = ((s_h > prev_hi) & (s_c < prev_hi)).astype(float)
        sweep_lo = ((s_l < prev_lo) & (s_c > prev_lo)).astype(float)
        brk_hi = (s_c > prev_hi).astype(float)
        brk_lo = (s_c < prev_lo).astype(float)
        # decaying memory of the most recent event (half-life ~5 bars)
        _add(F, f"sweep_{n}", ewm(sweep_lo - sweep_hi, 7), "sweep")
        _add(F, f"breakout_{n}", ewm(brk_hi - brk_lo, 7), "sweep")

    # ---------------- J. round-number levels (Osler 2003) -----------------
    px = P["s_close"].astype("float64")
    for step in (1000.0, 500.0, 100.0):
        below = np.floor(px / step) * step
        frac = (px - below) / step  # 0..1 position inside the grid cell
        dist_up = np.log((below + step) / px) / sig
        dist_dn = np.log(px / below.clip(lower=1.0)) / sig
        _add(F, f"rn{int(step)}_up", np.minimum(dist_up, 50), "round")
        _add(F, f"rn{int(step)}_dn", np.minimum(dist_dn, 50), "round")
        crossed = np.sign(np.floor(px / step) - np.floor(px.shift(5) / step))
        _add(F, f"rn{int(step)}_cross5", crossed, "round")

    # ---------------- K. same-phase (clock-cycle) lags --------------------
    for lag in (15, 30, 60):
        _add(F, f"phase_lag_{lag}", z1.shift(lag - 1), "phase")
    _add(F, "phase_hour_avg", (z1.shift(59) + z1.shift(119) + z1.shift(179)) / 3, "phase")

    # ---------------- M. regime state: how well each mechanism has worked LATELY ---------
    # The relationships themselves change (e.g. perp->spot lead-lag was ~0 during Binance's
    # zero-fee BTCUSDT period Jul-2022..Mar-2023 and jumped 5x when fees returned). Each
    # feature is a trailing correlation between a signal and the return that FOLLOWED it,
    # using only bars <= t, so the model can see which regime it is in. 1-3 day windows fit
    # in the live history (8000 bars).
    zc = z1.clip(-6, 6)
    bdev = pd.Series(F["basis_dev_60"], index=idx).clip(-6, 6)
    bfr = pd.Series(F["basis_fresh"], index=idx).clip(-6, 6)
    clv_s = pd.Series(F["clv"], index=idx)
    vdev = pd.Series(F["vwap_dev_30"], index=idx).clip(-6, 6)
    r5 = ((s_c - s_c.shift(5)) / (sig.shift(5) * np.sqrt(5))).clip(-6, 6)   # realised 5m move
    def rcorr(a, b, w):
        return a.rolling(w, min_periods=w // 2).corr(b)
    _add(F, "rg_leadlag_1d", rcorr(bdev.shift(1), zc, 1440), "regime")
    _add(F, "rg_leadlag_3d", rcorr(bdev.shift(1), zc, 4320), "regime")
    _add(F, "rg_fresh_1d", rcorr(bfr.shift(1), zc, 1440), "regime")
    _add(F, "rg_micromom_1d", rcorr(zc.shift(1), zc, 1440), "regime")
    _add(F, "rg_micromom_3d", rcorr(zc.shift(1), zc, 4320), "regime")
    _add(F, "rg_clv_1d", rcorr(clv_s.shift(1), zc, 1440), "regime")
    _add(F, "rg_meanrev_3d", rcorr(vdev.shift(5), r5, 4320), "regime")
    rv1 = r.rolling(1440, min_periods=720).std()
    rv5 = r.rolling(7200, min_periods=3600).std()
    _add(F, "rg_vol_1d_vs_5d", np.log(rv1 / rv5), "regime")
    _add(F, "rg_liq_1d_vs_5d", np.log((qv.rolling(1440, min_periods=720).mean() + 1) /
                                      (qv.rolling(7200, min_periods=3600).mean() + 1)), "regime")

    out = pd.DataFrame(F, index=idx)
    out = out.replace([np.inf, -np.inf], np.nan)
    if families is not None:
        out = out[[c for c in out.columns if FAMILY[c] in families]]
    return out, sig


def make_targets(P, sig, horizons=(1, 5), sec=None, delays=(0, 2, 5, 10, 30),
                 sec_horizons=(10, 15), sec_delays=(0, 2, 5)):
    """Forward log returns of spot close, raw and in sigma units.

    z{h}        : close(t) -> close(t+h)                 (theoretical, act at the close)
    zd{d}_{h}   : price(close(t)+d sec) -> close(t+h)     (what you capture acting d seconds late)
    zs{d}_{H}   : price(close(t)+d sec) -> price(close(t)+d+H sec)   (next H seconds after acting)
    `sec` is the frame produced by scripts/download_1s.py (px_d columns). Seconds targets are
    scaled by sigma * sqrt(H/60) so every horizon is in comparable volatility units."""
    s_c = np.log(P["s_close"].astype("float64"))
    T = {}
    for h in horizons:
        fr = s_c.shift(-h) - s_c
        T[f"r{h}"] = fr
        T[f"z{h}"] = fr / (sig * np.sqrt(h))
        if sec is not None:
            for d in delays:
                start = np.log(sec[f"px_{d}"].reindex(P.index))
                T[f"zd{d}_{h}"] = (s_c.shift(-h) - start) / (sig * np.sqrt(h))
    if sec is not None:
        for H in sec_horizons:
            for d in sec_delays:
                a, b = f"px_{d}", f"px_{d + H}"
                if a in sec and b in sec:
                    T[f"zs{d}_{H}"] = (np.log(sec[b].reindex(P.index)) - np.log(sec[a].reindex(P.index))) / (sig * np.sqrt(H / 60))
    return pd.DataFrame(T, index=P.index)
