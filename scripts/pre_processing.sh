#!/bin/bash
#SBATCH --job-name=brainsss
#SBATCH --partition=trc
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=./logs/mainlog.out
#SBATCH --open-mode=append
#SBATCH --mail-type=ALL

ml python/3.6.1
ml antspy/0.2.2
date
python3 -u /home/users/ykhao/registration/brain2vec/scripts/pre_processing.py &
wait
