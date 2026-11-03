# btc-predict2: a latency-aware Bitcoin predictor for the next 10 seconds to 5 minutes

At every 1-minute candle close this predicts, for the next 10 and 15 seconds and the next
1, 2, 3, 4 and 5 minutes:

* **P(up)**: the probability BTC/USDT will be higher than the price you can get *now*,
  where "now" means 2 seconds after the last 1-minute candle closed (when a bot can act);
* a **median move** and a **90% price range** from a separate volatility model;
* the **drivers** behind each call (per-feature contributions), so you can see *why* it leans
  up or down;
* for 10 and 15 seconds, also the chance the price **does not change at all** (13–18% of
  10-second windows).

The 10 s / 15 s forecasts cover the first seconds after each candle close, because the inputs
are 1-minute candles. Forecasting every 10 seconds all minute long would need second-by-second
perp data.

It is built from what holds up out of sample on 2021–2026 Binance data. Some well-known
ideas did not hold up; the tests behind every claim are under `research/`.

## Results on the sealed holdout (May–October 2026, never seen during research)

Scored once, by the locked model. "Act +2 s" means the move is measured from the price
2 seconds after the candle closed, which is when a bot can actually trade on the forecast.

| Horizon | Accuracy | AUC | Top-10% most confident | Top-1% | If you act at +0 s / +30 s |
|---|---|---|---|---|---|
| next 10 s ¹ | **56.49%** | 0.592 | **66.8%** | 70.5% | 59.0% / (+5 s) 54.5% |
| next 15 s ¹ | **56.06%** | 0.588 | **66.6%** | 69.7% | 58.3% / (+5 s) 54.3% |
| next 1 min | **53.95%** | 0.557 | **61.2%** | 66.4% | 54.9% / 51.3% |
| next 2 min | 53.21% | 0.546 | 59.0% | 64.1% | 53.8% / 51.5% |
| next 3 min | 52.67% | 0.540 | 57.7% | 62.1% | 53.1% / 51.4% |
| next 4 min | 52.54% | 0.538 | 57.6% | 60.4% | 52.8% / 51.5% |
| next 5 min | **52.44%** | 0.536 | **57.2%** | 60.3% | 52.7% / 51.6% |

¹ Added later. The design was chosen on the 2023 – April 2026 folds, then these targets were
scored once on the same holdout months. Accuracy counts only windows where the price moved
(it stayed flat in 18.6% of 10-second and 14.6% of 15-second windows).

* Every holdout month beat 52% at 1 minute (range 52.0–54.9%) and at 5 minutes (52.0–52.8%),
  and 53% at 10 and 15 seconds (53.4–58.1%).
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
* **Speed matters.** The 10-second edge falls from 59.0% acting at the close to 56.5% at +2 s
  and 54.5% at +5 s. The 1-minute edge falls from 54.9% to 51.3% acting 30 s late. The
  5-minute edge is far more robust (52.7% → 51.6%).
* The shortest horizons are the most accurate and the least profitable per call: a 10-second
  move is about 1–2 bp, so even 70% accuracy is worth about 0.4 bp.
* Markets change. The live loop scores every forecast it makes (`logs/live_predictions.csv`),
  so you can see whether the edge is still there.

## What actually moves Bitcoin over 10 seconds to 5 minutes

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

**10. At 10–15 seconds only the fast signals matter, and they are stronger there.** Rank IC
of the perp–spot gap, acting 2 s after the close: 10.6% for the next 10 s, 10.0% for 15 s,
7.6% for 1 minute. Less noise piles up in a shorter window. Mean reversion is about zero at
this scale. Regime features won 7/7 folds at 10 s and 5/7 at 15 s, and the range family was
dropped there (it won only 1/7). Both seconds models use 46 features. The price does not
move at all in 13–18% of 10-second windows since 2023 (about 25% when the market is calm,
against 2–4% in the zero-fee era), so the live tool reports that chance separately.
`research/02_latency/seconds_horizons.py` and `make research-seconds` hold the tests.

## Regime changes

The diagnosis comes first. All decisions below were made on the 2023 – April 2026 folds;
`research/06_regimes/` holds the tests.

**The edge survives ordinary regimes.** Walk-forward accuracy at 1 minute, by regime:

| Regime | 1-min accuracy |
|---|---|
| strong bear → strong bull (7-day trend) | 52.8% → 53.5%: trend does not matter |
| much calmer → much wilder than the last 90 days | 54.8% → 51.6%: weaker in turmoil, never negative |
| thin → heavy volume | 54.3% → 52.0% |
| model age, month 1 → month 6 after training | 53.4% → 53.2%: no real staleness |

Only 1 of 40 months had negative skill.

**Structural breaks are the real risk.** Signal strength measured month by month splits into
three kinds:
* **Mean reversion** has the same sign in 100% of months. Its strength varies about 2x and
  it never flips.
* **The perp lead** has the same sign in 95–98% of months, but its strength varies 5x.
* **Candle micro-shape** (close location, last-minute move) has the same sign in only
  58–77% of months, and it **flips**.

The break is sharp and has a known cause: **Binance's zero-fee BTCUSDT spot promotion
(July 2022 – March 22, 2023)**. With free arbitrage, spot tracked the perp almost instantly.
The perp-lead IC sat at 0–2% and the candle signals were reversed. Within weeks of fees
returning, the perp-lead IC jumped to 9–20% and the candle signals turned positive. The fast
edge is, in effect, the price of arbitrage.

**What was done about it.**

| Remedy | Result on 2023–26 folds | Decision |
|---|---|---|
| **Regime-state features** (1-min only): trailing 1–3-day strength of each mechanism (perp lead, gap freshness, minute momentum, close location, mean reversion), plus volatility and volume vs the past 5 days, all from past data only | 1 min: +4.1 bp log-loss, **6/7 folds**, biggest gain (+19 bp) right after the 2023 break. 2 min: 3/7, 3 min: 1/7, 5 min: 2/7 | **kept for 1 min only** |
| Recency-weighted training (6-month half-life) | 1 min 3/7, 5 min 2/7. It helps after a break and hurts otherwise | rejected |
| **Online recalibration**: each day the live loop scores itself on the last 7 days and refits the probability temperature | 1 min: +2.0 bp (folds), +0.7 (holdout), +9.7 (stress). 2–5 min: about neutral | **always on for 1 min; circuit breaker for 2–5 min** (engages only when the 7-day skill turns negative) |
| Monthly retraining | no measurable staleness over 6 months | optional (see below) |

**Stress test: what if zero-fee returns?** A model trained only on today's regime
(Apr 2023 – Mar 2025) predicted the zero-fee period (training on later data to test earlier
data is fine for robustness, but it is not a forecasting claim):

| 1-minute model | Log-loss skill | Top-10% accuracy |
|---|---|---|
| original design | **−12.7 bp** (worse than a coin flip) | 50.8% |
| + regime features | −5.4 bp | 52.1% |
| + online recalibration (full design) | **+4.3 bp** | 52.9% |

The 5-minute model needed no help: +17.9 bp in the zero-fee regime, because mean
reversion is structural.

**Regime-aware design on the holdout** (a second look, after the original single scoring):
1-minute log-loss skill rose from 47.7 to 52.8 bp, AUC from 0.557 to 0.558, and top-10% accuracy
from 61.2% to 61.6%. Horizons 2–5 are unchanged by design.

**Live behaviour.** At startup the predictor loads 2 weeks of history. It then:
* scores its own forecasts over the last 7 days, never on minutes the model trained on;
* refits temperatures, hourly;
* prints a regime line (volatility and volume vs the week, current perp-lead strength) and
  its 7-day track record;
* warns when any horizon's edge is not confirmed.

**First forward check (truly new data).** In the 2.6 days after the live model's training
cutoff (Nov 1–3, 2026, about 3,700 minutes), the self-check measured accuracy on close-to-close
outcomes of 55.3% (1 min), 52.7% (2 min), 52.3% (3 min), 51.8% (4 min) and 52.4% (5 min). The
sample is small but consistent with the holdout.

## Running it

```bash
make install
```

```bash
make live
```

The trained model ships in `models/live` (trained on data through 2026-10-31). Live forecasts
use public market data and need no API key. Spot data comes from `data-api.binance.vision`;
perp data comes from `fapi.binance.com`, or `www.binance.com/fapi` where that host is
geo-blocked. `make once` prints a single forecast at the next candle close, and `make test`
runs the offline tests (no lookahead, live parity, model sanity) in about 3 seconds.

Real output (`make live`, 2026-11-03):

```
  self-check, last 2.6 days: 1m hit 55.2% skill +81bp temp 1.21* | 2m hit 52.7% skill +33bp temp 1.13 | 3m hit 52.4% skill +16bp temp 1.07 | 4m hit 52.1% skill +7bp temp 1.14 | 5m hit 52.5% skill +26bp temp 1.22
  (* = temperature refitted on this self-check window; 1m always, other horizons only when their edge broke)

2026-11-03 14:56:01 UTC | BTC 77,346.00 | last candle 14:55 | data age 1.7s
  horizon  P(up)   confidence      median move   90% range
   10s      51.6%   15.7 pct          -0.00 bp   77,242 – 77,449   no change  5.1%
   15s      51.8%   20.0 pct          -0.00 bp   77,241 – 77,453   no change  3.2%
    1m      50.7%   11.6 pct          +0.00 bp   77,225 – 77,469
    2m      51.3%   27.4 pct          +0.01 bp   77,188 – 77,510
    3m      50.5%   11.9 pct          +0.03 bp   77,178 – 77,520
    4m      50.2%    5.3 pct          +0.03 bp   77,152 – 77,547
    5m      50.4%   10.2 pct          +0.08 bp   77,114 – 77,586
  (10s/15s: counted from ~2s after the candle close; P(up) is given that the price moves)
  why 10s: + perp-vs-spot gap (perp leads) (+0.17); + perp close vs its bar VWAP (+0.56); - volatility level (+8.26); + perp-vs-spot gap change (+0.20)
  why 1m: + perp-vs-spot gap (perp leads) (+0.17); - where the candle closed in its range (+0.64); - perp-lead strength, last 24h (+0.04); - volatility rising/falling (+0.13)
  why 5m: + perp-vs-spot gap (perp leads) (+0.17); + fresh perp-spot gap (opened late in the bar) (+0.09); + distance above 24h low (+0.40); - distance below 1h high (+0.29)
  regime: volatility normal for this week (-0.03) | volume normal | perp-lead last 24h normal (+0.041)
```

* `confidence` is the percentile of this call's conviction among the model's recent calls.
  Most minutes sit near 50%, which is correct: the model only leans when it has a reason.
* The `self-check` line is the model scoring itself on minutes it never trained on.
* After 10 or more forecasts, a `live scorecard` line reports the realised hit rate. The
  10 s / 15 s forecasts are scored one minute later against 1-second candles
  (`logs/live_seconds.csv`).

Rebuild everything from scratch (about 2 GB download; roughly an hour on 4 cores):

```bash
make data
```

```bash
make holdout
```

```bash
make calibrate
```

```bash
make model
```

Retraining on newly published months is optional (no staleness was measured over 6 months):

```bash
make data START=2026-11 END=2026-11
```

```bash
make model
```

`make help` lists every step. Research experiments need `make cache` first; see
`research/README.md`.

## Repository map

| Path | What it is |
|---|---|
| `AGENTS.md` | a complete guide for coding agents (also loaded by `CLAUDE.md`) |
| `btcpred/` | the library: `paths`, `data`, `features` (shared by research, training and live), `validate`, `model`, `live` |
| `scripts/` | the pipeline in run order: `download`, `download_1s`, `build_panel`, `train`, `calibrate`, `predict_live` |
| `config/` | the frozen design: `model_design.json` (features per horizon), `selected_features.json`, `calibration.json` |
| `models/live/` | the shipped model |
| `research/` | experiments grouped by question (`01_facts` … `06_regimes`), named feature sets in `sets/`, and an index |
| `reports/` | `holdout/` (scored reports: `v1_locked` is the single sealed look), `cv/` (per-fold metrics), `tables/`, `logs/` |
| `tests/` | offline tests on a 20,000-minute fixture |

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
