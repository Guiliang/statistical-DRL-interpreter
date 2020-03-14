#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --account=def-functor
cd /home/functor/scratch/xiangyus/LMUT
#virtualenv python2
source python2/bin/activate
cat requirements.txt | xargs -n 1 pip install
python main.py