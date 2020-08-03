#!/bin/bash
launch_time="2020-8-03"
aid=4
cpuct=0.005
game=SpaceInvaders-v0
log_dir="${game}-output-single-cput${cpuct}-play200-action${aid}-${launch_time}.out"
touch $log_dir
for ((n=0;n<31;n++))
do
    echo shell running round $n
    python3 run_mimic_learner.py -a $aid -r $n -d $log_dir -m mcts -g $game -l $launch_time 2>&1 &
    process_id=$!
    wait $process_id
    echo shell finish running round $n
done