#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --account=def-functor
#SBATCH --output=output-single-cput0_01-2020-3-25.out
cd /home/functor/scratch/Galen/project-DRL-Interpreter/statistical-DRL-interpreter/
#virtualenv python2
source sdl-venv-py37/bin/activate
cd /home/functor/scratch/Galen/project-DRL-Interpreter/statistical-DRL-interpreter/interface
python ./run_mimic_learner.py