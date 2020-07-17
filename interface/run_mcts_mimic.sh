#!/bin/bash
launch_time="2020-7-13"
aid=0
game=flappybird
log_dir="output-single-cput0_02-play1000-action${aid}-${launch_time}.out"
touch $log_dir
for ((n=9;n<31;n++))
do
    echo shell running round $n
    python3 run_mimic_learner.py -a $aid -r $n -d $log_dir -m mcts -g $game -l $launch_time 2>&1 &
    process_id=$!
    wait $process_id
    echo shell finish running round $n
done