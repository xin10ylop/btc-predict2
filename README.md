# btc-predict2: a latency-aware Bitcoin predictor for the next 1–5 minutes

Every minute this predicts, for horizons of 1, 2, 3, 4 and 5 minutes:

* **P(up)**: the probability BTC/USDT will be higher than the price you can get *now*,
  where "now" means 2 seconds after the last 1-minute candle closed (when a bot can act);
* a **median move** and a **90% price range** from a separate volatility model;
* the **drivers** behind each call (per-feature contributions), so you can see *why* it leans
  up or down.

It is built from what holds up out of sample on 2021–2026 Binance data. Some well-known
ideas did not hold up; the tests behind every claim are under `research/`.

> RESULTS_PLACEHOLDER

## What actually moves Bitcoin over 1–5 minutes

> FINDINGS_PLACEHOLDER

## Running it

```bash
pip install -r requirements.txt
```

Live forecasts use public market data and need no API key. Spot data comes from
`data-api.binance.vision`; perp data comes from `fapi.binance.com`, or `www.binance.com/fapi`
where that host is geo-blocked:

```bash
python scripts/predict_live.py
```

Rebuild everything from scratch (about 1.5 GB download, roughly 30 minutes on 4 cores):

```bash
python scripts/download.py --start 2021-01 --end 2026-10
```

```bash
python scripts/download_1s.py 2021-01 2026-10
```

```bash
python -m btcpred.data
```

```bash
python scripts/train.py --mode holdout
```

```bash
python scripts/train.py --mode live
```

## Method

**Data.** Binance BTCUSDT spot and USDT-M perpetual 1-minute klines, the perp premium-index
klines, ETHUSDT spot, and BTCUSDT spot 1-second klines, January 2021 to October 2026
(3.07 million minutes). All of it comes from `data.binance.vision`.

**Latency-aware targets.** Most published "next candle" predictors score the move from the
candle close. Nobody can trade at the exact close. The 1-second data gives the price d seconds
after every close, and the target is the move from that price to the close h minutes later.
The production model trains at d = 2 s and is also scored at 0, 10 and 30 s, which shows how
fast each signal dies.

**Scale-free features.** Every price feature is expressed in units of the running volatility,
and every volume feature relative to its own recent history. A $20k market in 2022 and a $120k
market in 2025 then look the same to the model. Binance fee changes moved raw volume and trade
counts severalfold between 2021 and 2026, and these features are immune to that.

**Validation.**
* Walk-forward: seven 6-month test folds from 2023-01 to 2026-04. Each model trains only on
  the 12, 24 or 36 months before its fold, with a purge gap.
* A feature family stays only if dropping it hurts out-of-sample log-loss on average and in at
  least 5 of the 7 folds, at either horizon. These rules were fixed before the results came in.
* **May–October 2026 was sealed.** No exploration, no feature choices and no tuning touched
  it, and it was scored exactly once, at the end.

**Model.**
* Direction: LightGBM binary classifiers per horizon, averaged over 12-, 24- and 36-month
  training windows. Strong regularisation: at least 3,000 rows per leaf, L2 penalty, feature and
  row subsampling.
* Distribution: a LightGBM scale model on log|move|, plus empirical residual quantiles.
* Probabilities are calibrated by construction (log-loss objective) and checked bin by bin.
