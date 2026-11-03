#!/bin/bash
cd /home/user/btc-predict2
while kill -0 "$1" 2>/dev/null; do sleep 5; done
export PYTHONPATH=.
python3 -W ignore research/vol_model.py 1 > reports/vol_h1.log 2>&1
python3 -W ignore research/vol_model.py 5 > reports/vol_h5.log 2>&1
echo QUEUE3_DONE >> reports/vol_h5.log
