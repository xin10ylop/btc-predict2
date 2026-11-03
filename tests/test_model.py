import numpy as np
import pandas as pd
import pytest
from btcpred import paths
from btcpred.model import BTCPredictor, HORIZONS


@pytest.fixture(scope="module")
def pred():
    return BTCPredictor.load(paths.LIVE_MODEL)


def test_predictions_are_valid(pred, features):
    X, _ = features
    F = pred.predict_frame(X.tail(500))
    for h in HORIZONS:
        p = F[f"p_up_{h}"]
        assert p.between(0.2, 0.8).all(), f"h={h}: implausible probability"
        assert (F[f"scale_{h}"] > 0).all()


def test_describe_bands_are_ordered(pred, features):
    X, sig = features
    d = pred.describe(X.iloc[[-1]], 77000.0, float(sig.iloc[-1]))
    for h in HORIZONS:
        b = d[h]["band_bp"]
        assert b["q5"] < b["q25"] <= b["q50"] <= b["q75"] < b["q95"], (h, b)
        assert 0 <= d[h]["confidence_pct"] <= 100


def test_self_check_never_uses_training_minutes(pred, panel, features):
    """The fixture (April 2026) is inside the live model's training period: no self-check."""
    X, _ = features
    p = BTCPredictor.load(paths.LIVE_MODEL)
    assert pd.Timestamp(p.meta["trained_until"]) > X.index[-1]
    p.online_calibrate(X, panel["s_close"].astype(float))
    assert p.track == {}


def test_self_check_runs_after_cutoff(panel, features):
    X, _ = features
    p = BTCPredictor.load(paths.LIVE_MODEL)
    p.meta["trained_until"] = str(X.index[-1] - pd.Timedelta(days=3))   # pretend: 3 days of new data
    tr = p.online_calibrate(X, panel["s_close"].astype(float))
    assert set(tr) == set(HORIZONS)
    for h, t in tr.items():
        assert 0.0 <= t["temp"] <= 3.0 and t["n"] > 2000 and 0.4 < t["hit"] < 0.7
