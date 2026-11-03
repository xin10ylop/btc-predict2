#!/bin/bash
# Waits for a given PID (the first runner) then runs the clock-reversion comparison.
cd /home/user/btc-predict2
while kill -0 "$1" 2>/dev/null; do sleep 5; done
export PYTHONPATH=. FEATVER=_v3
python3 -W ignore research/compare_sets.py 5 research/sets_clockrev.json > reports/clockrev_h5.log 2>&1
python3 -W ignore research/compare_sets.py 1 research/sets_clockrev.json > reports/clockrev_h1.log 2>&1
echo QUEUE2_DONE >> reports/clockrev_h1.log
