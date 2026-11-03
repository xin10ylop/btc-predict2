# btc-predict2: guide for agents

You are working on a Bitcoin predictor. At every 1-minute candle close it gives, for the next
10 s, 15 s, 1, 2, 3, 4 and 5 minutes:
* P(up) for Binance BTCUSDT spot,
* a median move and a 90% price range,
* the features driving the call,
* for 10 s and 15 s, the chance the price does not change at all.

It is built on 2021–2026 Binance data and validated walk-forward on a sealed holdout. This
file is everything you need to work on it safely. Read it fully before changing anything.

## Read in this order

1. This file.
2. `README.md`: results, findings and the regime analysis, with numbers.
3. `research/README.md`: which script proves which claim, plus the selection rule.
4. `btcpred/features.py` and `btcpred/model.py`, which are the core.

## Map

```
btcpred/            the library (pip install -e .)
  paths.py          every file location; import paths, never hard-code "data/..."
  data.py           Binance CSV loading, panel building, loaders (load_panel, load_sec_offsets, load_dev)
  features.py       ALL features + targets; one function shared by research, training and live
  validate.py       walk-forward folds, metrics, HOLDOUT_START = 2026-05-01
  model.py          BTCPredictor: per-horizon ensemble, scale model, bands, calibration, online self-check
  live.py           public Binance endpoints with host fallbacks
scripts/            the pipeline, in run order: download -> download_1s -> build_panel -> train -> calibrate -> predict_live
config/             the frozen design: model_design.json (features per horizon), calibration.json
models/live/        the shipped model (trained through 2026-10-31), gzip LightGBM text files
research/           experiments grouped by question (01_facts ... 06_regimes), sets/ = named feature lists
reports/            results: holdout/ (scored reports), cv/ (per-fold metrics), tables/, logs/
tests/              offline tests on a 20,000-minute fixture (no network, ~3 s)
data/               git-ignored; rebuilt by `make data` and `make cache`
```

## Commands (`make help`)

`make install` · `make test` · `make live` · `make once` · `make data` · `make cache` ·
`make holdout` · `make calibrate` · `make model` · `make research-regimes`

## How it works

* **Data.** 1-minute klines for BTCUSDT spot (`s_*`), the USDT-M perp (`p_*`), the perp
  premium index (`pi_*`) and ETHUSDT (`e_*`). Spot 1-second klines give the price d seconds
  after each close (`data/sec`).
* **Row convention.** Row t is the candle that *opened* at t. Features at row t use only
  candles up to and including t, so they are known at t's close.
* **Targets.** `zd{d}_{h}` runs from the price d seconds after the close of t to the close of
  t+h, divided by the running volatility. `zs{d}_{H}` runs from close+d s to close+d+H s, for
  H = 10, 15. **Production trains on d = 2** (when a bot can act).
* **Horizon keys** in code are `"s10"`, `"s15"` (seconds) and `1`..`5` (minutes). Use the
  helpers `is_sec`, `h_minutes`, `h_label` and `target_col` in `model.py`; never assume
  horizons are ints.
* **Features** are scale-free: prices in units of the running volatility, volumes relative to
  their own history. Each feature has a *family* (`FAMILY` in `features.py`).
* **Model.** Per horizon: LightGBM binary classifiers trained on the last 12, 24 and 36 months
  and averaged; a LightGBM scale model; empirical residual quantiles for bands. Feature sets
  (`config/model_design.json`):
  * 1 min: the 47 base features + 9 regime-state features;
  * 10 s / 15 s: 46 "fast" features (base + regime, without the slow range family);
  * 2–5 min: the 47 base features.
* **Seconds horizons** also hold a no-change table: P(price unchanged) by decile of predicted
  volatility (`flat_tab`).
* **Calibration.** Averaging makes probabilities timid. The static temperatures and band
  widths in `config/calibration.json` were fitted on the holdout after it was scored.
* **Live** (`scripts/predict_live.py`): loads 20,000 minutes of history. It self-scores on up
  to 7 days *after the training cutoff* and refits the temperature, hourly: always at 1 min,
  and at 2–5 min only when the recent edge turned negative (circuit breaker). 10 s / 15 s keep
  their static calibration and are scored one minute later against 1-second candles
  (`logs/live_seconds.csv`). Each forecast fires about 1.5 s after the candle close.

## Rules you must not break

1. **No lookahead.** A feature at t may use only data up to t. `tests/test_features.py::test_no_lookahead`
   must pass. Add any new feature to `features.py` (with a family) so the test covers it.
2. **Latency-aware evaluation.** Score with `zd2` (and report `zd0`, `zd10`, `zd30`). Never
   claim accuracy measured from the exact close as tradeable.
3. **The holdout (2026-05-01 to 2026-10-31) is used up.** It was scored once by the locked
   model, then looked at a second time after the regime work. Do not tune on it. Make new
   decisions on the dev folds (`validate.TEST_FOLDS`). The only untouched data is data after
   2026-10-31, which the live self-check measures.
4. **Selection rule.** Keep a change only if it beats the reference on log-loss on average
   **and in at least 5 of 7 folds** at the horizon where it is used. Write it down before
   running. Do not bend it after seeing results.
5. **Live parity.** Live computes features on 20,000 minutes. Any new feature must settle
   within about 8,000 minutes (no windows longer than 5 days). `test_live_warmup_parity` checks
   this.
6. **One LightGBM job at a time.** Parallel 4-thread jobs on 4 cores run more than 5x slower.
7. **Never commit `data/`.** `models/live/` is committed; regenerate it with `make model`.
   `config/` is the single source of truth for the shipped design.

## What is already known (don't redo it)

Works, with the same sign in every year:
* the perp–spot gap (perp leads spot);
* **gap freshness** (close-basis minus bar-VWAP basis, recovered from kline volume fields);
* mean reversion: 30-minute VWAP stretch, distance from the 1-hour high or low, 15–60-minute
  returns;
* breakouts fail and sweeps reverse (but these are redundant);
* clock effects;
* the ETH lead, which is small.

Measured:
* **Fast signals decay within seconds, slow ones are flat out to 30 s.**
* Lead-lag shows only in calm markets.
* Flow-driven moves revert and flow-free moves do not.
* VWAP reversion is 2x stronger before quarter-hours and 3–4x stronger at funding settlement
  since 2023. This is not in the model (4/7 folds); the live tool raises an alert instead.

Rejected:
* round numbers (they hurt);
* the global spot index or "Coinbase premium" lead (none);
* buyer vs seller VWAP;
* same-phase lags;
* raw flow, impact and sweep families (redundant);
* recency-weighted training;
* regime features at 2–5 min;
* 94 features (the 47 win).

Regimes:
* robust across trend, volatility and volume regimes;
* no staleness over 6 months;
* **the structural break was Binance's zero-fee BTCUSDT period (Jul 2022 – Mar 2023)**, when
  the fast edge vanished. It is handled by the regime features plus the online self-check
  (stress test: −12.7 bp → +4.3 bp at 1 min).

Holdout results, acting 2 s after the close:
* 10 s: 56.5% accuracy on windows where the price moves (66.8% top-10%); 59.0% acting at the
  close, 54.5% acting 5 s late;
* 15 s: 56.1% (66.6% top-10%);
* 1 min: 53.95% accuracy (61.6% on the top-10% most confident calls);
* 5 min: 52.44% (57.2% top-10%);
* every month above 52%.

**Economics.** Confident calls happen in calm markets, so the edge is under about 1 bp per
call, below taker fees. The value is in direction-only payoffs and in timing orders you would
place anyway.

## Recipes

* **Add a feature.**
  1. Add it in `features.py` with a family tag.
  2. Run `make cache`.
  3. Write `research/sets/<x>.json` with the reference `core47` and core47 plus the new feature.
  4. Run `python research/04_selection/compare_sets.py 1 research/sets/<x>.json`, then the
     same with `5`.
  5. Apply the rule on `reports/cv/*.csv`.
  6. If it passes, add it to `config/model_design.json`, then run `make model` and `make test`.
* **Retrain on new months.**
  1. Run `make data START=<first new month> END=<last full month>` (downloads are cached).
  2. Run `make model`, then `make test`.
  3. Commit `models/live`.
* **Investigate live behaviour.** Forecasts are logged to `logs/live_predictions.csv`.
  `BTCPredictor.online_calibrate` holds the self-check; `scripts/predict_live.py` holds the
  display and the regime line.

## Gotchas

* Since 2025-01-01, Binance spot files use microsecond timestamps; `data.py` and the 1-second
  downloader handle it.
* `api.binance.com` and `fapi.binance.com` are geo-blocked in some countries. `live.py` falls
  back to `data-api.binance.vision` (spot) and `www.binance.com/fapi` (perp).
* pandas 3 can return read-only arrays from `.values`, so avoid in-place `&=` on them.
* Pipe long jobs with `python -u`. `pgrep -f <pattern>` can match its own shell, so kill by
  PID.
* The 1-minute edge depends on speed, and the live loop is built to act about 1.5 s after
  the close. Don't add slow work before the forecast.
