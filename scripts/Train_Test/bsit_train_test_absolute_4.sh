#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --nodes=3
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=11:59:59
#SBATCH --output=../outputs/%x-%a.out

source ../../env2/bin/activate
cd ../src && python -W ignore main.py --dataset 4 --exp_name bsit_train_test_absolute --model bsit --epochs 50 --batch_size 200 --frequency_bands all --minimal_data_needed --runs 3 --absolute_values 1
