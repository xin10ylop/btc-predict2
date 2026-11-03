# Reports

Committed results. Every file here was produced by a script in `scripts/` or `research/`.
Accuracy is measured acting 2 s after the candle close unless a file says otherwise.

## `holdout/`: scores on May–October 2026

These are the months no research decision was made on. Files are in the order they were
produced:

| File | Model | Horizons | What it is |
|---|---|---|---|
| `v1_locked.json` / `.log` | original design | 1–5 min | **the single sealed look**: scored once, before anything else touched these months |
| `v2_regime_aware.json` / `.log` | + regime features at 1 min | 1–5 min | second look, after the regime work (designed on dev folds) |
| `v2_regime_aware_online.json` | same, replaying the live 7-day self-check | 1–5 min | the effect of online recalibration |
| `v3_seconds.json` / `.log` | 10 s / 15 s models (46 fast features) | 10 s, 15 s | first look at these targets; same months as above |

Each JSON has, per horizon:
* `delay_<d>s`: metrics when acting d seconds late (`acc`, `auc`, `ll_skill_bp`, `ic`,
  `acc_top10`, `edge_top10`);
* `selective`: accuracy and edge in bp when keeping only the most confident x% of calls;
* `monthly_acc`;
* `calibration`: predicted vs realised P(up) by bin;
* `band90_coverage`;
* for seconds horizons, `flat_share` (no price change);
* for 5 min, `aligned_5m_windows_acc`.

A new `scripts/train.py --mode holdout` run writes `latest.json` and `latest_online.json`.

## `cv/`: walk-forward results on the development period

`<set>_h<h>.csv` holds the per-fold metrics (7 folds, 2023-01 to 2026-04) of feature set
`<set>` (`research/sets/`) at horizon `<h>` (`1`..`5`, `s10`, `s15`). To compare two sets,
difference their `ll_skill_bp` columns. The selection rule is in `research/README.md`.

## `tables/`

* `ablation_h1.csv`: drop-one-family ablation at 1 min.
* `monthly_ic.csv`: signal strength month by month (the regime study).

## `logs/`

Console output of the experiments and of the live-model training (`train_live.log`).
