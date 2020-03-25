#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=4
#SBATCH --account=def-functor
cd /home/functor/scratch/Galen/project-DRL-Interpreter/statistical-DRL-interpreter/interface
#virtualenv python2
source sdl-venv-py37/bin/activate
python ./run_mimic_learner.py