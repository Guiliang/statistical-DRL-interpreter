#!/bin/bash
#SBATCH --time=60:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=20
#SBATCH --account=def-functor
#SBATCH --output=output-single-cput0_01-2020-8-02.out
cd /home/functor/scratch/Galen/project-DRL-Interpreter/statistical-DRL-interpreter/
#virtualenv python3
source sdl-venv-py37/bin/activate
cd /home/functor/scratch/Galen/project-DRL-Interpreter/statistical-DRL-interpreter/interface
nohup ./run_mcts_mimic.sh >/dev/null 2>log &