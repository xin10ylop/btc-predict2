# btc-predict2: every step of the project, in order. `make help` lists targets.
PY ?= python3
START ?= 2021-01
END   ?= 2026-10

.PHONY: help install test live once data panel cache holdout calibrate model research-regimes research-seconds clean-cache

help:          ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

install:       ## install the package (editable) and test deps
	$(PY) -m pip install -e ".[dev]"

test:          ## offline tests: no lookahead, live parity, model sanity (~3 s)
	$(PY) -m pytest -q

live:          ## live forecasts every minute (public Binance data, no key)
	$(PY) -u scripts/predict_live.py

once:          ## one forecast at the next candle close
	$(PY) -u scripts/predict_live.py --once

data:          ## download 1m klines + 1s offsets (~2 GB, ~10 min) and build the panel
	$(PY) scripts/download.py --start $(START) --end $(END)
	$(PY) scripts/download_1s.py $(START) $(END)
	$(PY) scripts/build_panel.py

panel:         ## rebuild data/panel.parquet from downloaded CSVs
	$(PY) scripts/build_panel.py

cache:         ## research cache: all features + targets on the dev period (data/cache/)
	$(PY) research/build_cache.py

holdout:       ## train < 2026-05-01 and score May-Oct 2026 -> reports/holdout/latest*.json
	$(PY) scripts/train.py --mode holdout

calibrate:     ## fit static temperatures/band widths on the holdout -> config/calibration.json
	$(PY) scripts/calibrate.py models/holdout 2026-05-01 2026-11-01

model:         ## train the shipped live model on all data -> models/live
	$(PY) scripts/train.py --mode live

research-regimes: ## re-run the regime study (drift, regime performance, stress tests)
	$(PY) research/06_regimes/regime_drift.py
	$(PY) research/06_regimes/regime_perf.py
	$(PY) research/06_regimes/stress_zero_fee.py
	$(PY) research/06_regimes/stress_online.py

research-seconds: ## re-run the 10s/15s study (signals, then feature sets; ~15 min)
	$(PY) research/02_latency/seconds_horizons.py
	$(PY) research/04_selection/compare_sets.py s10 research/sets/seconds.json
	$(PY) research/04_selection/compare_sets.py s15 research/sets/seconds.json

clean-cache:   ## delete the research cache (keeps raw data and the panel)
	rm -rf data/cache
