"""Fit per-horizon probability temperature and band width on a period the model did not
train on (used once, on the sealed holdout, AFTER it was scored). Writes reports/calibration.json.

    python scripts/calibrate.py models/holdout 2026-05-01 2026-11-01
"""
import json, os, sys, glob
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from btcpred.model import BTCPredictor, HORIZONS, QGRID   # noqa: E402
from btcpred.features import make_features, make_targets   # noqa: E402

folder, a, b = sys.argv[1], sys.argv[2], sys.argv[3]
pred = BTCPredictor.load(folder)
pred.calib = {}
P = pd.read_parquet("data/panel.parquet")
X, sig = make_features(P)
S = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/sec/*.parquet"))])
S = S[~S.index.duplicated()].sort_index()
T = make_targets(P, sig, horizons=HORIZONS, sec=S, delays=(pred.delay,))
m = (X.index >= pd.Timestamp(a, tz="UTC")) & (X.index < pd.Timestamp(b, tz="UTC"))
F = pred.predict_frame(X[m])
out = {}
for h in HORIZONS:
    z = T[f"zd{pred.delay}_{h}"][m].values
    ok = np.isfinite(z) & (z != 0)
    p = np.clip(F[f"p_up_{h}"].values[ok], 1e-6, 1 - 1e-6)
    y = (z[ok] > 0).astype(float)
    lg = np.log(p / (1 - p))
    nll = lambda t: -np.mean(y * np.log(1 / (1 + np.exp(-t * lg))) + (1 - y) * np.log(1 - 1 / (1 + np.exp(-t * lg))))
    temp = float(minimize_scalar(nll, bounds=(0.5, 3.0), method="bounded").x)
    # band width so that the 5-95% band covers 90% of outcomes
    pc = 1 / (1 + np.exp(-temp * lg))
    Q = np.array(pred.resid_q[h]); sc = F[f"scale_{h}"].values[ok]
    delta = -np.interp(1 - pc, QGRID, Q)
    lo = np.interp(.05, QGRID, Q) + delta; hi = np.interp(.95, QGRID, Q) + delta
    cov = lambda k: np.mean((z[ok] >= lo * sc * k) & (z[ok] <= hi * sc * k))
    ks = np.linspace(0.8, 1.5, 71); band_k = float(ks[np.argmin([abs(cov(k) - 0.90) for k in ks])])
    out[h] = {"temp": round(temp, 4), "band_k": round(band_k, 4)}
    print(h, out[h], f"nll gain {(nll(1.0) - nll(temp)) * 1e4:.2f}bp  coverage {cov(1.0)*100:.1f}% -> {cov(band_k)*100:.1f}%")
json.dump({str(k): v for k, v in out.items()}, open("reports/calibration.json", "w"), indent=1)
