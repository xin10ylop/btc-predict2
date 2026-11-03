# btc-predict2: guide for agents

## The prompt

Paste this to any agent that works on the repo:

> You are working on btc-predict2, a Bitcoin direction predictor for the next 10 s, 15 s and
> 1–5 minutes, built on Binance data. Before changing anything, read `AGENTS.md` fully, then
> `README.md` (results and findings), `research/README.md` (which script proves which claim)
> and `reports/README.md` (what each result file is). Run `make install` and then `make test`.
> Follow the rules in `AGENTS.md`:
> * no lookahead;
> * score from 2 s after the candle close;
> * never tune on the May–October 2026 holdout;
> * keep a change only if it wins at least 5 of 7 walk-forward folds;
> * run one training job at a time;
> * run `make test` after every change.
>
> If the docs and the code disagree, trust the code and fix the docs.

Everything below is the reference behind that prompt.

## 1. What it does

At every 1-minute candle close it predicts, for **10 s, 15 s, 1, 2, 3, 4 and 5 minutes**
ahead:
* P(up) for Binance BTCUSDT spot,
* a median move and a 90% price range,
* the features behind the call,
* for 10 s and 15 s, the chance the price does not change at all.

"Ahead" is counted from 2 seconds after the close, when a bot can act. Results on the
holdout (May–Oct 2026), acting 2 s after the close:

| Horizon | Accuracy | Most confident 10% |
|---|---|---|
| 10 s | 56.5% (on windows where the price moves) | 66.8% |
| 15 s | 56.1% | 66.6% |
| 1 min | 54.0% | 61.6% |
| 5 min | 52.4% | 57.2% |

Every month was above 52%. The edge per call is under about 1 bp, below taker fees. The
value is in direction-only payoffs and in timing orders you would place anyway.

## 2. Map

```
btcpred/          the library (make install = pip install -e .)
  paths.py        every file location. Import it; never hard-code "data/..."
  data.py         load Binance CSVs, build the panel; loaders: load_panel, load_sec_offsets, load_dev
  features.py     ALL features and targets, shared by research, training and live
  validate.py     walk-forward folds, metrics, HOLDOUT_START = 2026-05-01
  model.py        BTCPredictor: per-horizon models, bands, no-change table, calibration, self-check
  live.py         public Binance endpoints with host fallbacks; 1-second prices
scripts/          the pipeline, in order: download, download_1s, build_panel, train, calibrate, predict_live
config/           the frozen design: model_design.json (features per horizon), calibration.json
models/live/      the shipped model, trained through 2026-10-31
research/         experiments by question (01_facts … 06_regimes); sets/ holds named feature lists; README is the index
reports/          results: holdout/, cv/ (per fold), tables/, logs/; README explains each file
tests/            offline tests on a 20,000-minute fixture (about 3 s, no network)
data/, logs/      git-ignored; rebuilt by `make data` / `make cache`, written by the live loop
```

**Commands** (`make help`):
* `make install`, `make test`
* `make live` (a forecast every minute), `make once` (one forecast)
* `make data`, `make cache`
* `make holdout`, `make calibrate`, `make model`
* `make research-regimes`, `make research-seconds`

## 3. How it works

1. **Data.** 1-minute candles for BTCUSDT spot (`s_*`), the USDT-M perp (`p_*`), the perp
   premium index (`pi_*`) and ETHUSDT (`e_*`). Spot 1-second candles give the price at
   +0 … +30 s after every close (`data/sec`, columns `px_<d>`).
2. **Rows.** Row t is the candle that *opened* at t. Its features use only candles up to and
   including t.
3. **Features** are scale-free: prices in units of the running volatility, volumes relative to
   their own history. Each has a family (`FAMILY` in `features.py`).
4. **Targets**, in volatility units:
   * `zd{d}_{h}`: from close+d s to the close h minutes later;
   * `zs{d}_{H}`: from close+d s to close+d+H s, for H = 10, 15;
   * production trains on **d = 2**.
5. **Horizon keys** in code are `"s10"`, `"s15"` and `1`..`5`. Use `is_sec`, `h_minutes`,
   `h_label` and `target_col` from `model.py`; never assume horizons are ints.
6. **Models** (per horizon):
   * LightGBM classifiers trained on the last 12, 24 and 36 months, then averaged;
   * a LightGBM volatility model plus residual quantiles, for the price bands;
   * for seconds horizons, a no-change table: P(price unchanged) by volatility decile.
7. **Feature sets** (`config/model_design.json`):
   * 1 min: 47 base features + 9 regime-state features;
   * 10 s / 15 s: 46 fast features (the 1-minute set without the slow range family);
   * 2–5 min: the 47 base features.
8. **Calibration.** Averaging makes probabilities timid. `config/calibration.json` holds a
   temperature and a band width per horizon, fitted on the holdout after it was scored.
9. **Live** (`scripts/predict_live.py`) loads 20,000 minutes of history, then:
   * scores itself on the minutes *after* the training cutoff (up to 7 days) and refits
     temperatures, hourly: always at 1 min, and at 2–5 min only if the recent edge turned
     negative;
   * keeps the static calibration for 10 s / 15 s, and checks those forecasts a minute later
     against 1-second candles (`logs/live_seconds.csv`);
   * forecasts about 1.5 s after each close.

## 4. Rules

1. **No lookahead.** `tests/test_features.py::test_no_lookahead` must pass. Every new feature
   goes in `features.py` with a family, so the test covers it.
2. **Score latency-aware.** Use `zd2` / `zs2`; report `zd0`/`zd10`/`zd30` and `zs0`/`zs5`
   too. Accuracy measured from the exact close is not tradeable.
3. **The holdout (2026-05-01 to 2026-10-31) is used up.** It was looked at three times (see
   `reports/README.md`). Never tune on it. Decide on the dev folds (`validate.TEST_FOLDS`). The
   only untouched data comes after 2026-10-31, and the live self-check measures it.
4. **Selection rule.** Keep a change only if it beats the reference on log-loss on average
   **and in at least 5 of 7 folds** at the horizon where it is used. Write the test down
   before running it, and do not bend it afterwards.
5. **Live parity.** Features must settle within about 8,000 minutes (no window longer than 5
   days). `test_live_warmup_parity` checks this.
6. **One LightGBM job at a time.** Two parallel 4-thread jobs on 4 cores run more than 5x
   slower.
7. **Never commit `data/`.** `config/` is the single source of truth for the design;
   `models/live/` is rebuilt with `make model` and committed.

## 5. What is already known (don't redo it)

**Works**, with the same sign every year:
* the perp–spot gap (the perp leads spot);
* gap freshness (close basis minus bar-VWAP basis, from kline volume fields);
* mean reversion: 30-minute VWAP stretch, distance from the 1-hour high or low, 15–60-minute
  moves;
* small effects: clock (minute and hour), bar shape, ETH.

**Measured:**
* Fast signals die within seconds; slow ones are flat out to 30 s.
* At 10–15 s only fast signals matter, and they are stronger there (perp–spot IC 10.6% at
  10 s vs 7.6% at 1 min).
* Lead-lag shows only in calm markets.
* Flow-driven moves revert; flow-free ones do not.
* The price is unchanged in 13–18% of 10 s windows (up to 43% when very quiet).
* VWAP reversion is 2x stronger before quarter-hours and 3–4x at funding settlement. It is
  not in the model (4/7 folds); the live tool shows an alert instead.

**Rejected:**
* round numbers;
* the global spot index / Coinbase-premium lead;
* buyer vs seller VWAP;
* same-phase lags;
* raw flow, impact and sweep families (redundant);
* recency-weighted training;
* regime features at 2–5 min;
* the range family at 10 s / 15 s;
* 94 features (47 win).

**Regimes:**
* robust across trend, volatility and volume;
* no staleness over 6 months;
* the one structural break was Binance's zero-fee BTCUSDT period (Jul 2022 – Mar 2023),
  when the fast edge vanished. Regime features plus the self-check handle it (1 min stress
  test: −12.7 bp → +4.3 bp).

## 6. Recipes

* **Add a feature.**
  1. Add it to `features.py` with a family.
  2. Run `make cache`.
  3. Write `research/sets/<x>.json` with the reference set and the reference plus your
     feature.
  4. Run `python research/04_selection/compare_sets.py <h> research/sets/<x>.json` for each
     horizon it should help (`s10`, `1`, `5`, ...).
  5. Apply the rule to `reports/cv/*.csv`.
  6. If it passes, add it to `config/model_design.json`, then run `make model` and `make test`.
* **Add a horizon.**
  1. Add a target in `make_targets`.
  2. Add a key in `model.py` (`MIN_HORIZONS` / `SEC_HORIZONS`).
  3. Pick its feature set with `compare_sets.py`.
  4. Score it with `train.py --mode holdout --horizons <h>`.
  5. Run `calibrate.py`, `make model` and `make test`.
* **Retrain on new months.** Run `make data START=<month> END=<month>`, then `make model`
  and `make test`, then commit `models/live`.
* **Debug live.** Forecasts go to `logs/live_predictions.csv` and seconds outcomes to
  `logs/live_seconds.csv`. The self-check is `BTCPredictor.online_calibrate`; the display
  is in `scripts/predict_live.py`.

## 7. Gotchas

* Since 2025-01-01, Binance spot files use microsecond timestamps; the loaders handle it.
* `api.binance.com` / `fapi.binance.com` are geo-blocked in some countries. `live.py` falls
  back to `data-api.binance.vision` and `www.binance.com/fapi`.
* pandas 3 can return read-only arrays from `.values`, so don't use in-place `&=` on them.
* Run long jobs with `python -u`. `pgrep -f` can match its own shell, so kill by PID.
* The short horizons depend on speed. Don't add slow work before the forecast in the live
  loop.
