#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --nodes=1
#SBATCH --gres=gpu:k80:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=11:59:59
#SBATCH --array=0-4
#SBATCH --output=../outputs/%x-%a.out

source ../../env2/bin/activate
cd ../src && python -W ignore main.py --dataset $SLURM_ARRAY_TASK_ID --exp_name rf_theta_frequency_bands --model rf --frequency_bands theta --riemann_type frequency
