"""btcpred: latency- and regime-aware Bitcoin direction predictor for the next 10 seconds to 5 minutes.

Modules
    paths     every file location (absolute)
    data      Binance CSV loading, panel building, cache loaders
    features  all features + targets; shared by research, training and live (no lookahead)
    validate  walk-forward folds, metrics, the sealed-holdout date
    model     BTCPredictor: per-horizon temporal-ensemble LightGBM, scale model, bands,
              calibration, online self-recalibration
    live      public Binance endpoints with host fallbacks
"""
__version__ = "1.1.0"
