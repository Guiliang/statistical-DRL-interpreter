#!/bin/bash
launch_time="2020-8-10"
aid=4
cpuct=0
play=2
game=SpaceInvaders-v0
log_dir="${game}-output-single-cput${cpuct}-play${play}-action${aid}-${launch_time}.out"
touch $log_dir
for ((n=31;n<201;n++))
do
    echo shell running round $n
    python3 run_mimic_learner.py -c $cpuct -p $play -a $aid -r $n -d $log_dir -m mcts -g $game -l $launch_time 2>&1 &
    process_id=$!
    wait $process_id
    echo shell finish running round $n
done