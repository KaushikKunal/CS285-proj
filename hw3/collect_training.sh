#!/bin/bash

for a in 0 0.1 0.2 0.3 0.5 0.7
do
for b in 0 0.1 0.2 0.3 0.5 0.8
do
python3 measure_energy.py python3 cs285/scripts/run_hw3_dqn.py \
    -cfg experiments/dqn/lunarlander_doubleq.yaml  \
    -neeval 100 -lra $a -pa $b --poll 1 -load
done
done