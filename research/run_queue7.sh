#!/bin/bash
cd /home/user/btc-predict2
export PYTHONPATH=.
python3 -W ignore scripts/train.py --mode holdout > reports/holdout_v2.log 2>&1
python3 -W ignore research/stress_online.py > reports/stress_online.log 2>&1
echo QUEUE7_DONE >> reports/stress_online.log
