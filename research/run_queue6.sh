#!/bin/bash
cd /home/user/btc-predict2
while kill -0 "$1" 2>/dev/null; do sleep 5; done
export PYTHONPATH=. FEATVER=_v4
python3 -W ignore research/compare_sets.py 2 research/sets_regime_mid.json > reports/regime_h2.log 2>&1
python3 -W ignore research/compare_sets.py 3 research/sets_regime_mid.json > reports/regime_h3.log 2>&1
echo QUEUE6_DONE >> reports/regime_h3.log
