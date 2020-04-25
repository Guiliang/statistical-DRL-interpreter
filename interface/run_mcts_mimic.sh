#!/bin/bash
log_dir="output-single-cput0_01-play1000-2020-4-22.out"
touch $log_dir
for ((n=9;n<30;n++))
do
    echo shell running round $n
    python3 run_mimic_learner.py -r $n -d $log_dir -m mcts -g flappybird 2>&1 &
    process_id=$!
    wait $process_id
    echo shell finish running round $n
done