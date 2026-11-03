import json
import numpy as np
from btcpred import paths
from btcpred.features import make_features


def model_features():
    d = json.load(open(paths.DESIGN))
    return sorted(set(d["base"]).union(*d["hfeatures"].values()))


def test_no_lookahead(panel):
    """Changing the future must not change any feature in the past."""
    k = len(panel) - 300
    fut = panel.copy()
    num = [c for c in fut.columns if c != "pi_close"]
    fut.iloc[k:, fut.columns.get_indexer(num)] *= 1.03       # shock every price/volume after row k
    a, _ = make_features(panel)
    b, _ = make_features(fut)
    diff = (a.iloc[:k] - b.iloc[:k]).abs().max().max()
    assert diff == 0, f"feature at t depends on data after t (max diff {diff})"


def test_every_configured_feature_is_computed(features):
    X, _ = features
    last = X.iloc[-1]
    for f in model_features():
        assert f in X.columns, f
        assert np.isfinite(last[f]), f"{f} is not finite on the latest bar"


def test_live_warmup_parity(panel, features):
    """The live loop keeps a finite history; features must match full-history values."""
    full, _ = features
    short, _ = make_features(panel.tail(12000))
    cols = model_features()
    a, b = short[cols].iloc[-200:], full[cols].iloc[-200:]
    rel = ((a - b).abs() / (b.abs().mean() + 1e-6)).max().max()
    assert rel < 1e-3, f"warm-up too short: max normalised diff {rel:.2e}"
