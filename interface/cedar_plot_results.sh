#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --account=def-functor
cd /home/functor/scratch/Galen/project-DRL-Interpreter/statistical-DRL-interpreter/
#virtualenv python3
source sdl-venv-py37/bin/activate
cat requirements.txt | xargs -n 1 pip install
cd /home/functor/scratch/Galen/project-DRL-Interpreter/statistical-DRL-interpreter/interface
log_dir="log_cedar_run_plot_results.out"
python run_plot_results.py -d $log_dir