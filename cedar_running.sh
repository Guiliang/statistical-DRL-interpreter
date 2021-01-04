#!/bin/bash
#SBATCH --time=200:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10
#SBATCH --account=def-functor
cd /home/functor/scratch/Galen/project-DRL-Interpreter/statistical-DRL-interpreter/
# virtualenv python3
source sdl-venv-py37/bin/activate
# cat requirements.txt | xargs -n 1 pip install
cd /home/functor/scratch/Galen/project-DRL-Interpreter/statistical-DRL-interpreter/interface
chmod 777 run_mcts_mimic.sh
./run_mcts_mimic.sh