#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=20
#SBATCH --account=def-functor
#SBATCH --output=output-single-cput0_001-2020-8-03.out
cd /home/functor/scratch/Galen/project-DRL-Interpreter/statistical-DRL-interpreter/
#virtualenv python3
source sdl-venv-py37/bin/activate
cat requirements.txt | xargs -n 1 pip install
cd /home/functor/scratch/Galen/project-DRL-Interpreter/statistical-DRL-interpreter/interface
nohup ./run_mcts_mimic.sh >/dev/null 2>log &