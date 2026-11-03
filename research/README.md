# Research index

Every claim in the top-level README comes from a script here. Run from anywhere after
`make install`. Most scripts need the research cache: `make data` once, then `make cache`.
All of them use the development period only (before the sealed holdout, 2026-05-01).

Conventions:
* `h` is the horizon in minutes.
* `zd{d}_{h}` is the return from d seconds after the candle close to the close h minutes
  later, divided by the running volatility. The production target is `zd2`.
* **IC** is the rank correlation between a feature and the future return, in %.
* **Log-loss skill** (bp) is how much the predicted probabilities beat the base rate.
* A **fold** is one of seven 6-month walk-forward test periods, 2023-01 to 2026-04.

| # | Question | Script | Output | Answer |
|---|---|---|---|---|
| 0 | build the cache | `build_cache.py` | `data/cache/` | all 106 features, all targets |
| 1 | How big and how autocorrelated are 1m moves? Clock effects? | `01_facts/stylized_facts.py` | stdout | small momentum in normal moves; :00 drifts up and :01/:31 drift down |
| 1 | Do big moves revert and small ones continue? | `01_facts/conditional_moves.py` | stdout | yes; the sign is stable by year |
| 1 | Which single features predict, every year? | `01_facts/univariate_ic.py` | stdout | perp–spot gap (+7.7%); VWAP stretch and 30m move revert (−4%) |
| 2 | Does the edge survive acting late? (ticks) | `02_latency/latency_decay.py` | stdout | the first view, from 3 sample days, said gone within 5 s. That was wrong; see next row |
| 2 | Which signals are fast or slow? (1 s data, 3 yrs) | `02_latency/delay_ic.py` | stdout | lead-lag decays over seconds; mean reversion is flat to 30 s |
| 3 | Is the gap freshness from kline VWAP new information? | `03_signals/intrabar_ic.py` | stdout | `basis_fresh` IC +6.8%, 0.34 correlated with the gap; buyer/seller VWAP useless |
| 3 | Does the global spot index (Coinbase etc.) lead? | `03_signals/index_gap_ic.py` | stdout | no (IC ≈ 0); the perp leads |
| 3 | Do flow-driven moves revert? | `03_signals/impact_test.py` | stdout | yes (t −4.6 to −6.5 each year); flow-free moves do not |
| 3 | Where do edges concentrate (volatility, clock, funding)? | `03_signals/clock_vol_conditioning.py` | stdout | lead-lag only when calm; reversion 2x before quarter-hours, 3–4x at funding (2023+) |
| 4 | First baseline | `04_selection/baseline.py <h>` | `data/cache/pred/base_h*.parquet` | 1m 54.1% acc acting at the close |
| 4 | Which feature families matter? | `04_selection/ablation.py 1` | `reports/tables/ablation_h1.csv` | basis +15 bp in 7/7 folds; round numbers hurt |
| 4 | Compare named feature sets | `04_selection/compare_sets.py <h> sets/<x>.json` | `reports/cv/<set>_h<h>.csv` | core47 beats the full 94 at 1m and 5m |
| 4 | Single window or ensemble? Huber? | `04_selection/model_variants.py <h> sel` | stdout | the 12/24/36-month ensemble wins |
| 5 | Is confidence worth money? | `05_economics/selective.py <pred.parquet> <h> [d]` | stdout | top 1%: 66–72% acc, but under 1 bp edge |
| 5 | Is volatility predictable beyond EWMA? | `05_economics/vol_model.py <h>` | `reports/logs/vol_h1.log` | Spearman 0.22–0.28; bands cover about 88–90% |
| 6 | Do relationships drift? | `06_regimes/regime_drift.py` | `reports/tables/monthly_ic.csv` | break at the end of zero fees (Mar 2023); mean reversion never flips |
| 6 | Where does the model fail? Does it go stale? | `06_regimes/regime_perf.py` | stdout | robust to trend; weaker in wild markets; no staleness |
| 6 | Does online recalibration help? | `06_regimes/online_recal.py` | stdout | 7-day window best: +2.0 bp (1m), +1.3 bp (5m) |
| 6 | Does live (close-to-close) recalibration work? | `06_regimes/online_recal_proxy.py` | stdout | yes, with adjustment factors 0.87–0.95 |
| 6 | What if zero fees return? | `06_regimes/stress_zero_fee.py` | `reports/logs/stress_zero_fee.log` | 1m −12.7 bp → −5.4 bp with regime features; 5m fine |
| 6 | Full design under stress | `06_regimes/stress_online.py` | `reports/logs/stress_online.log` | 1m +4.3 bp with online recalibration |

## Feature sets (`sets/`)

| File | Sets | Used for |
|---|---|---|
| `baseline.json` | `baseline_62` | the first model |
| `ablation.json` | `ablation_universe` (90) | drop-one ablation |
| `v2.json` | `v1_all` (90), `v2_all` (94, + intra-bar) | the intra-bar test |
| `grouped.json` | `core`, `core47`, `core_impact`, `core_sweep`, `core_slowall` | which slow family to keep |
| `clockrev.json` | `core47`, `core_range_clockrev` | clock-reversion test (4/7, rejected) |
| `regime.json` / `regime_mid.json` | `core_regime` (56) vs `core47` | regime features: kept for 1m only |
| `recency.json` | `core_recency6` (run with `HALFLIFE=6`) | recency weighting (rejected) |

## The selection rule (fixed before testing)

A feature family or design change is kept only if it beats the reference on out-of-sample
log-loss on average **and** in at least **5 of 7 folds**, at the horizon where it is used.
Compare by differencing the `ll_skill_bp` columns of two `reports/cv/*.csv` files.

## Practical notes

* Run **one LightGBM job at a time**. Two 4-thread jobs on 4 cores thrash OpenMP and run
  more than 5x slower.
* A walk-forward run takes about 2–3 minutes per set and horizon with 47 features.
* Use `python -u` (or `flush=True`) when piping long jobs, or output stays buffered.
