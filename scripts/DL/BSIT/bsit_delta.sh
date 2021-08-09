#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --nodes=2
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=7:59:59
#SBATCH --array=0-4
#SBATCH --output=../outputs/%x-%a.out

source ../../env2/bin/activate
cd ../src && python -W ignore main.py --dataset $SLURM_ARRAY_TASK_ID --exp_name bsit_delta --model bsit --epochs 50 --frequency_bands delta
