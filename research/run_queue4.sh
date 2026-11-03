#!/bin/bash
cd /home/user/btc-predict2
while kill -0 "$1" 2>/dev/null; do sleep 5; done
export PYTHONPATH=.
python3 -W ignore research/model_variants.py 5 sel > reports/variants_h5.log 2>&1
python3 -W ignore research/model_variants.py 1 sel > reports/variants_h1.log 2>&1
echo QUEUE4_DONE >> reports/variants_h1.log
