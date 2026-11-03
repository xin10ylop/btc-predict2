#!/bin/bash
cd /home/user/btc-predict2
while pgrep -f "research/ablation.py" > /dev/null; do sleep 5; done
export PYTHONPATH=.
python3 -W ignore research/compare_sets.py 1 research/sets_v2.json > reports/compare_v2_h1.log 2>&1
python3 -W ignore research/compare_sets.py 5 > reports/compare_v2_h5.log 2>&1
echo QUEUE1_DONE >> reports/compare_v2_h5.log
