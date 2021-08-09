#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --nodes=1
#SBATCH --gres=gpu:k80:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=11:59:59
#SBATCH --array=0-5
#SBATCH --output=../outputs/%x-%a.out

source ../../env2/bin/activate
cd ../src && python -W ignore main.py --dataset $SLURM_ARRAY_TASK_ID --exp_name svm_delta_frequency_bands --model svm --frequency_bands delta --svm_kernel linear --riemann_type frequency
