# btc-predict2: a latency-aware Bitcoin predictor for the next 1–5 minutes

Every minute this predicts, for horizons of 1, 2, 3, 4 and 5 minutes:

* **P(up)**: the probability BTC/USDT will be higher than the price you can get *now*,
  where "now" means 2 seconds after the last 1-minute candle closed (when a bot can act);
* a **median move** and a **90% price range** from a separate volatility model;
* the **drivers** behind each call (per-feature contributions), so you can see *why* it leans
  up or down.

It is built from what holds up out of sample on 2021–2026 Binance data. Some well-known
ideas did not hold up; the tests behind every claim are under `research/`.

## Results on the sealed holdout (May–October 2026, never seen during research)

Scored once, by the locked model. "Act +2 s" means the move is measured from the price
2 seconds after the candle closed, which is when a bot can actually trade on the forecast.

| Horizon | Accuracy | AUC | Top-10% most confident | Top-1% | If you act at +0 s / +30 s |
|---|---|---|---|---|---|
| next 1 min | **53.95%** | 0.557 | **61.2%** | 66.4% | 54.9% / 51.3% |
| next 2 min | 53.21% | 0.546 | 59.0% | 64.1% | 53.8% / 51.5% |
| next 3 min | 52.67% | 0.540 | 57.7% | 62.1% | 53.1% / 51.4% |
| next 4 min | 52.54% | 0.538 | 57.6% | 60.4% | 52.8% / 51.5% |
| next 5 min | **52.44%** | 0.536 | **57.2%** | 60.3% | 52.7% / 51.6% |

* Every holdout month beat 52% at 1 minute (range 52.0–54.9%) and at 5 minutes (52.0–52.8%).
* The holdout matched or beat the 2023–2026 walk-forward (1 min: 53.3%, 5 min: 52.5%), so
  there is no sign of overfitting.
* At 5-minute-aligned windows (:00, :05, ..., the "up or down in 5 minutes" setup) the
  accuracy was 52.3%.
* The 90% price band covered 87–90% of outcomes. The live model widens it 2–11% and
  sharpens the probabilities, which the ensemble made too timid (predicted 61% → realised
  67%), with one temperature per horizon. Both were fitted on the holdout only after it had
  been scored; accuracy and rankings are unchanged by either.

### Read this before using it for money

* **Accuracy is not profit.** The most confident calls come in calm markets, where moves
  are tiny. At 66% accuracy the average 1-minute edge is about 0.55 basis points; at 5 minutes
  it is about 1.1 bp for the top 1%. Binance taker fees are 4–10 bp. **This is not a
  stand-alone taker strategy.**
* The edge is worth most where the payoff depends only on direction (up/down prediction
  markets), or for **timing an order you were going to place anyway**: buy in the
  minute the model expects to be cheaper.
* **Speed matters.** The 1-minute edge falls from 54.9% acting at the close to 51.3% acting
  30 s late. The 5-minute edge is far more robust (52.7% → 51.6%).
* Markets change. The live loop scores every forecast it makes (`logs/live_predictions.csv`),
  so you can see whether the edge is still there.

## What actually moves Bitcoin over 1–5 minutes

What follows comes from walk-forward tests on 2021–2026 data. A feature family stays only
if dropping it hurts out of sample in most folds. Scripts are in `research/`.

**1. Price discovery happens in the Binance perpetual. Spot follows.** The perp–spot gap
(vs its own last hour) is the strongest 1-minute predictor (rank IC +7.7%, the same sign in
all 6 years). Dropping it costs 15 bp of log-loss, in 7/7 folds. Spot closes the gap over
seconds to about a minute.
*Negative result:* the multi-exchange spot index (Coinbase, Kraken, OKX, ...) does **not**
lead Binance spot (IC ≈ 0). The "Coinbase premium" idea has no value at the minute scale.

**2. New: the *freshness* of the perp–spot gap, recovered from candle volume fields.**
Binance klines carry `quote_volume / volume`, which is the bar's VWAP. The gap at the close
minus the gap at the bar VWAP tells whether the dislocation opened late in the minute, before
spot could react. It scores IC +6.8% on its own and is only 0.34 correlated with the plain
gap. With the related close-vs-bar-VWAP features it improved the model in **7/7 folds**. The
symmetric idea, buyer VWAP minus seller VWAP, carries nothing (IC ≈ 0).

**3. Two kinds of signal: fast and slow.** Scored against the price *d* seconds after the
close (from 1-second data):

| Feature → target (rank IC) | act +0 s | +2 s | +5 s | +10 s | +30 s |
|---|---|---|---|---|---|
| perp–spot gap → next 1 min (fast) | 10.1% | 7.5% | 6.0% | 4.6% | 2.4% |
| distance below 1h high → next 5 min (slow) | 3.6% | 3.6% | 3.6% | 3.6% | 3.5% |

Lead-lag signals die within seconds. Mean-reversion signals barely notice a 30-second delay.

**4. Lead-lag is a calm-market phenomenon.** The perp–spot IC is 15.9% in the calmest
volatility quartile and 0.7% in the wildest. In violent markets everything moves together.
In quiet ones the perp's small leads stand out above the noise. Accuracy follows: about 57%
in low volatility against 52% in high.

**5. Over 1–60 minutes, stretched prices revert.** Distance above the 30-minute VWAP, distance
from the 1-hour high or low, and the last 15–60 minutes of return all predict the opposite
direction, consistently every year. Closing above the 1-hour high (a breakout) predicts
*down* over the next minutes. Pokes through a high or low that close back inside ("liquidity
sweeps") predict a reversal. Both are real, but redundant once the distance-to-range features
are present.

**6. Flow-driven moves revert; moves without flow stick.** Regress each return on aggressive
order flow, then split the last 15 minutes into a flow-explained part and a flow-free part.
The flow-explained part reverts (t = −4.6 to −6.5 in every full year). The flow-free part
mostly does not (t ≈ −0.5 to −1.6 in 2023–25). This matches transient-impact (propagator)
theory: market orders push price temporarily, while quote repositioning carries the
information.

**7. The clock leaves a footprint.**
* The first minute after :00 drifted up in 2021–25, and :01 and :31 drift down every year
  (top-of-hour recurring buys, then reversal).
* Deviations from VWAP revert about **2x harder before quarter-hours** in every year.
* Within ±1 minute of the 8-hourly funding settlement they revert **3–4x harder since 2023**
  (IC −9 to −17% against −3% normally), and the effect survives a 30-second delay.
* Settlement minutes are only 0.6% of the data, so as a model feature this passed only 4/7
  folds and was left out by the pre-registered rule. The live loop shows it as an alert.

**8. Tested and rejected.**
* Round-number levels ($100/$500/$1000; Osler-style stop clustering) made the model *worse*.
* Quarter-hour "same-phase" lags added nothing.
* Raw taker-flow imbalances are redundant once price features are present.
* ETH adds a little (ETH-vs-BTC last minute, IC +2.4%).
* **47 features beat all 94.** Every redundant family added noise.

**9. Training on several history lengths beats picking one.** Averaging models trained on
the last 12, 24 and 36 months beat each single window at both horizons, in log-loss and in
the worst fold.

## Running it

```bash
pip install -r requirements.txt
```

The trained models ship in `models/live` (trained on data through 2026-10-31). Live forecasts
use public market data and need no API key. Spot data comes from
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
python scripts/calibrate.py models/holdout 2026-05-01 2026-11-01
```

```bash
python scripts/train.py --mode live
```

Real output from the live loop on 2026-11-03 (abridged to one minute):

```
2026-11-03 12:45:01 UTC | BTC 77,388.01 | last candle 12:44 | data age 1.8s
  horizon  P(up)   confidence      median move   90% range
   1m      56.1%   80.7 pct top 25%   +0.22 bp   77,312 – 77,469
   2m      52.5%   49.8 pct          +0.09 bp   77,296 – 77,484
   3m      49.8%    5.4 pct          -0.00 bp   77,283 – 77,497
   4m      49.6%   10.8 pct          -0.03 bp   77,267 – 77,512
   5m      49.9%    3.0 pct          -0.01 bp   77,267 – 77,513
  why 1m: + perp-vs-spot gap (perp leads) (+0.27); + fresh perp-spot gap (opened late in the bar) (+0.26); - volatility level (+4.79); - premium level (-3.22)
  why 5m: + perp-vs-spot gap (perp leads) (+0.27); + fresh perp-spot gap (opened late in the bar) (+0.26); + perp-vs-spot gap change (+0.19); + 5m move (-0.96)
```

`confidence` is the percentile of this call's conviction among the model's recent calls.
Most minutes are near 50%, and that is correct: the model only leans when it has a reason.
After 10 or more forecasts, a `live scorecard` line reports the realised hit rate.

## Repository map

| Path | What it is |
|---|---|
| `btcpred/features.py` | every feature, shared by research, training and live (parity tested to <0.03%) |
| `btcpred/model.py` | `BTCPredictor`: temporal-ensemble direction models, scale model, price bands |
| `btcpred/live.py` | public-endpoint data access with host fallbacks |
| `btcpred/validate.py` | walk-forward harness and metrics |
| `scripts/` | download, train, calibrate, live loop |
| `research/` | every experiment behind the findings (IC scans, latency decay, ablations, regressions) |
| `reports/` | logs, per-fold CSVs, `holdout_report.json`, `calibration.json`, `selected_features.json` |

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
