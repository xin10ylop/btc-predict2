#!/bin/bash
cd /home/user/btc-predict2
export PYTHONPATH=. FEATVER=_v4
python3 -W ignore research/stress_zero_fee.py > reports/stress_zero_fee.log 2>&1
python3 -W ignore research/compare_sets.py 1 research/sets_regime.json > reports/regime_h1.log 2>&1
python3 -W ignore research/compare_sets.py 5 research/sets_regime.json > reports/regime_h5.log 2>&1
HALFLIFE=6 python3 -W ignore research/compare_sets.py 1 research/sets_recency.json > reports/recency_h1.log 2>&1
HALFLIFE=6 python3 -W ignore research/compare_sets.py 5 research/sets_recency.json > reports/recency_h5.log 2>&1
echo QUEUE5_DONE >> reports/recency_h5.log
