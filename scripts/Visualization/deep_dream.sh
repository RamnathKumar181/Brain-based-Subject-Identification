#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --nodes=2
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=8:59:59
#SBATCH --array=0-4
#SBATCH --output=../outputs/%x-%a.out

source ../../env2/bin/activate
cd ../src && python -W ignore main.py --dataset $SLURM_ARRAY_TASK_ID --exp_name deep_dream --vis --vis_type deep_dream
