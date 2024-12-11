#!/bin/bash
#
#SBATCH --job-name=explore
#
#SBATCH --time=1-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=512M

export PYTHONPATH="$PATH:$PWD:$1"

conda activate exploration
python exp_sac_minigrid.py --config_file $2 --seed $3 --arguments env_name $4 save_path $5