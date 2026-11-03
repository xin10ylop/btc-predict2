#!/bin/bash
cd /home/user/btc-predict2
while pgrep -f "queue1.sh" > /dev/null; do sleep 5; done
export PYTHONPATH=.
python3 -W ignore research/compare_sets.py 5 research/sets_grouped.json > reports/grouped_h5.log 2>&1
python3 -W ignore research/compare_sets.py 1 research/sets_grouped.json > reports/grouped_h1.log 2>&1
echo QUEUE2_DONE >> reports/grouped_h1.log
