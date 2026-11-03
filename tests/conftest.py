import pathlib, sys
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def panel():
    """20,000 real minutes (2026-04-10..04-23) with the live column layout."""
    return pd.read_parquet(ROOT / "tests" / "fixtures" / "panel_sample.parquet")


@pytest.fixture(scope="session")
def features(panel):
    from btcpred.features import make_features
    return make_features(panel)
