"""Train the production predictor.

    python scripts/train.py --mode holdout   # train < 2026-05-01, score ONCE on 2026-05..10 (sealed)
    python scripts/train.py --mode live      # train on everything, save models/live

Requires data/panel.parquet (btcpred.data) and data/sec/*.parquet (scripts/download_1s.py).
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from btcpred.features import make_features, make_targets, FAMILY   # noqa: E402
from btcpred.model import BTCPredictor, HORIZONS                    # noqa: E402
from btcpred.validate import metrics, HOLDOUT_START                 # noqa: E402

DELAYS = (0, 2, 5, 10, 30)


def load_all():
    P = pd.read_parquet("data/panel.parquet")
    X, sig = make_features(P)
    S = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/sec/*.parquet"))])
    S = S[~S.index.duplicated()].sort_index()
    T = make_targets(P, sig, horizons=HORIZONS, sec=S, delays=DELAYS)
    return P, X, T, sig


def evaluate(pred, X, T, start, end, out_json):
    msk = (X.index >= start) & (X.index < end)
    Xh, Th = X[msk], T[msk]
    F = pred.predict_frame(Xh)
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
    ap.add_argument("--features", default="reports/selected_features.json")
    ap.add_argument("--windows", default="12,24,36")
    a = ap.parse_args()
    P, X, T, sig = load_all()
    feats = json.load(open(a.features)) if os.path.exists(a.features) else list(X.columns)
    windows = tuple(int(w) for w in a.windows.split(","))
    pred = BTCPredictor(feats, delay=2, windows=windows)
    if a.mode == "holdout":
        pred.fit(X, T, HOLDOUT_START)
        pred.save("models/holdout")
        evaluate(pred, X, T, pd.Timestamp(HOLDOUT_START, tz="UTC"), X.index[-1] + pd.Timedelta(minutes=1),
                 "reports/holdout_report.json")
    else:
        pred.fit(X, T, X.index[-1] - pd.Timedelta(minutes=10))
        pred.save("models/live")
        print("saved models/live")


if __name__ == "__main__":
    main()
