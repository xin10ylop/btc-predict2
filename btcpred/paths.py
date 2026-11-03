"""Every file location in the project, in one place. All paths are absolute, so scripts work
from any working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# data/ (git-ignored, rebuilt by `make data` and `make cache`)
DATA = ROOT / "data"
RAW = DATA / "raw"                    # Binance CSVs per dataset (scripts/download.py)
TICKS = RAW / "ticks"                 # sample aggTrades days (research/02_latency)
SEC = DATA / "sec"                    # price d seconds after each minute close (scripts/download_1s.py)
PANEL = DATA / "panel.parquet"        # aligned 1-minute panel, 2021-01..2026-10 (scripts/build_panel.py)
CACHE = DATA / "cache"                # research cache (research/build_cache.py)
X_DEV = CACHE / "X_dev.parquet"       # all features, development period (< holdout)
T_DEV = CACHE / "T_dev.parquet"       # all targets, development period
FAMILIES = CACHE / "families.csv"     # feature -> family
PRED = CACHE / "pred"                 # out-of-sample walk-forward predictions, <set>_h<h>.parquet

# config/ (committed: the frozen production design)
CONFIG = ROOT / "config"
DESIGN = CONFIG / "model_design.json"         # per-horizon feature lists
SELECTED = CONFIG / "selected_features.json"  # the 47 base features
CALIBRATION = CONFIG / "calibration.json"     # static temperature / band width per horizon

# models/
MODELS = ROOT / "models"
LIVE_MODEL = MODELS / "live"          # committed, trained through 2026-10-31
HOLDOUT_MODEL = MODELS / "holdout"    # git-ignored, trained < 2026-05-01

# reports/ (committed results)
REPORTS = ROOT / "reports"
HOLDOUT_REPORTS = REPORTS / "holdout"
CV = REPORTS / "cv"                   # per-fold walk-forward metrics, <set>_h<h>.csv
TABLES = REPORTS / "tables"
LOGS = REPORTS / "logs"

# research/
SETS = ROOT / "research" / "sets"     # named feature sets, <name>.json -> {set: [features]}

# logs/ (git-ignored, live loop)
LIVE_LOG = ROOT / "logs" / "live_predictions.csv"


def ensure(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
