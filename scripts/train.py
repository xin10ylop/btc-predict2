"""Train the production predictor.

    python scripts/train.py --mode holdout   # train < 2026-05-01, score on 2026-05..10 -> reports/holdout/latest*.json
    python scripts/train.py --mode live      # train on everything -> models/live (applies config/calibration.json)

The design (per-horizon feature lists) comes from config/model_design.json.
Requires data/panel.parquet (scripts/build_panel.py) and data/sec/ (scripts/download_1s.py).
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from btcpred.features import make_features, make_targets, FAMILY   # noqa: E402
from btcpred.model import BTCPredictor, HORIZONS                    # noqa: E402
from btcpred.validate import metrics, HOLDOUT_START                 # noqa: E402
from btcpred import paths                                           # noqa: E402
from btcpred.data import load_panel, load_sec_offsets               # noqa: E402

DELAYS = (0, 2, 5, 10, 30)


def load_all():
    P = load_panel()
    X, sig = make_features(P)
    T = make_targets(P, sig, horizons=HORIZONS, sec=load_sec_offsets(), delays=DELAYS)
    return P, X, T, sig


def online_recal_sim(pred, F, Xh, close, days=7):
    """Replay the live loop's daily self-recalibration over the evaluation period: each day's
    temperature is fitted on the previous `days` days of close-to-close outcomes only."""
    from scipy.optimize import minimize_scalar
    out = F.copy()
    day = Xh.index.floor("D"); udays = day.unique()
    for h in HORIZONS:
        p = np.clip(F[f"p_up_{h}"].values, 1e-6, 1 - 1e-6); lg = np.log(p / (1 - p))
        c0 = close.reindex(Xh.index).values; c1 = close.shift(-h).reindex(Xh.index).values
        ok = np.isfinite(c0) & np.isfinite(c1) & (c0 != c1); y = (c1 > c0).astype(float)
        a_t = np.ones(len(p))
        for i, d in enumerate(udays):
            if i < days:
                continue
            rows = np.flatnonzero((day >= udays[i - days]) & (day < d) & ok)
            f = lambda a: -np.mean(y[rows] * np.log(np.clip(1 / (1 + np.exp(-a * lg[rows])), 1e-6, 1)) +
                                   (1 - y[rows]) * np.log(np.clip(1 - 1 / (1 + np.exp(-a * lg[rows])), 1e-6, 1)))
            a_t[day == d] = minimize_scalar(f, bounds=(0.0, 3.0), method="bounded").x * pred.PROXY_ADJ[h]
        out[f"p_up_{h}"] = 1 / (1 + np.exp(-a_t * lg))
    return out


def evaluate(pred, X, T, start, end, out_json, F=None):
    msk = (X.index >= start) & (X.index < end)
    Xh, Th = X[msk], T[msk]
    F = pred.predict_frame(Xh) if F is None else F
    rep = {"period": [str(start), str(end)], "horizons": {}}
    for h in HORIZONS:
        hr = {}
        p = F[f"p_up_{h}"].values
        for d in DELAYS:
            z = Th[f"zd{d}_{h}"].values
            ok = np.isfinite(z) & (z != 0)
            m = metrics(p[ok], (z[ok] > 0).astype(int), z[ok])
            hr[f"delay_{d}s"] = {k: float(v) for k, v in m.items()}
        # selective accuracy at the production delay
        z = Th[f"zd{pred.delay}_{h}"].values
        ok = np.isfinite(z) & (z != 0)
        conf = np.abs(p[ok] - 0.5)
        hit = (p[ok] > 0.5) == (z[ok] > 0)
        raw_bp = z[ok] * Xh["sig_bp"].values[ok] * np.sqrt(h)
        side = np.sign(p[ok] - 0.5)
        sel = {}
        for cov in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01):
            mm = conf >= np.quantile(conf, 1 - cov)
            sel[f"{int(cov*100)}%"] = dict(acc=float(hit[mm].mean()), edge_bp=float(np.mean(side[mm] * raw_bp[mm])),
                                          n=int(mm.sum()))
        hr["selective"] = sel
        # month-by-month stability
        mon = pd.Series(hit, index=Xh.index[ok]).groupby(Xh.index[ok].strftime("%Y-%m")).mean()
        hr["monthly_acc"] = {k: float(v) for k, v in mon.items()}
        # calibration
        bins = pd.cut(p[ok], [0, .4, .45, .48, .5, .52, .55, .6, 1])
        cal = pd.DataFrame({"p": p[ok], "y": z[ok] > 0, "b": bins}).groupby("b", observed=True).agg(
            pred=("p", "mean"), real=("y", "mean"), n=("y", "size"))
        hr["calibration"] = {str(k): {c: float(v) for c, v in r.items()} for k, r in cal.iterrows()}
        # band coverage of the distribution model (5%..95%)
        sc = F[f"scale_{h}"].values[ok]
        Q = np.array(pred.resid_q[h])
        delta = -np.interp(1 - p[ok], np.linspace(.01, .99, 99), Q)
        lo = (np.interp(.05, np.linspace(.01, .99, 99), Q) + delta) * sc
        hi = (np.interp(.95, np.linspace(.01, .99, 99), Q) + delta) * sc
        hr["band90_coverage"] = float(np.mean((z[ok] >= lo) & (z[ok] <= hi)))
        if h == 5:
            al = (Xh.index[ok].minute % 5) == 4
            hr["aligned_5m_windows_acc"] = float(hit[al].mean())
        rep["horizons"][h] = hr
        d2 = hr[f"delay_{pred.delay}s"]
        print(f"h={h}m acc={d2['acc']*100:.2f}% auc={d2['auc']:.4f} llskill={d2['ll_skill_bp']:.1f}bp "
              f"top10%acc={sel['10%']['acc']*100:.2f}% top1%acc={sel['1%']['acc']*100:.2f}% "
              f"band90cov={hr['band90_coverage']*100:.1f}% | acc by delay: " +
              " ".join(f"{d}s:{hr[f'delay_{d}s']['acc']*100:.2f}" for d in DELAYS), flush=True)
    json.dump(rep, open(out_json, "w"), indent=1)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["holdout", "live"], required=True)
    ap.add_argument("--design", default=str(paths.DESIGN))
    ap.add_argument("--windows", default="12,24,36")
    a = ap.parse_args()
    P, X, T, sig = load_all()
    design = json.load(open(a.design))
    windows = tuple(int(w) for w in a.windows.split(","))
    pred = BTCPredictor(design["base"], delay=2, windows=windows, hfeatures=design.get("hfeatures"),
                        halflife=design.get("halflife"))
    if a.mode == "holdout":
        pred.fit(X, T, HOLDOUT_START)
        pred.save(paths.HOLDOUT_MODEL)
        paths.ensure(paths.HOLDOUT_REPORTS)
        a0, b0 = pd.Timestamp(HOLDOUT_START, tz="UTC"), X.index[-1] + pd.Timedelta(minutes=1)
        print("--- static (raw ensemble) ---")
        evaluate(pred, X, T, a0, b0, paths.HOLDOUT_REPORTS / "latest.json")
        print("--- with live-style online recalibration (7-day self-check) ---")
        msk = (X.index >= a0) & (X.index < b0)
        Fo = online_recal_sim(pred, pred.predict_frame(X[msk], raw=True), X[msk], P["s_close"])
        evaluate(pred, X, T, a0, b0, paths.HOLDOUT_REPORTS / "latest_online.json", F=Fo)
    else:
        pred.fit(X, T, X.index[-1] - pd.Timedelta(minutes=10))
        if paths.CALIBRATION.exists():  # fitted on the holdout after it was scored
            pred.calib = {int(k): v for k, v in json.load(open(paths.CALIBRATION)).items()}
        pred.save(paths.LIVE_MODEL)
        print(f"saved {paths.LIVE_MODEL}")


if __name__ == "__main__":
    main()
