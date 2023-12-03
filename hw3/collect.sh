#!/bin/bash

for a in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
for b in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
python3 measure_energy.py python3 cs285/scripts/eval_hw3_dqn.py \
    -cfg experiments/dqn/lunarlander_doubleq.yaml  \
    -neval 100 -lra $a -pa $b --poll 1
done
done