#!/bin/bash
# Sequential experiment runner: one LightGBM job at a time (OpenMP oversubscription is catastrophic).
cd /home/user/btc-predict2
export PYTHONPATH=.
python3 -W ignore research/compare_sets.py 1 research/sets_v2.json > reports/compare_v2_h1.log 2>&1
python3 -W ignore research/compare_sets.py 5 > reports/compare_v2_h5.log 2>&1
python3 -W ignore research/compare_sets.py 5 research/sets_grouped.json > reports/grouped_h5.log 2>&1
python3 -W ignore research/compare_sets.py 1 research/sets_grouped.json > reports/grouped_h1.log 2>&1
echo QUEUE_DONE >> reports/grouped_h1.log
