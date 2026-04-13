#!/bin/bash
#SBATCH --job-name=mctd
#SBATCH --output=/data/user_data/mbronars/slurm/output/data_collection_%j.out
#SBATCH --error=/data/user_data/mbronars/slurm/error/data_collection_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=general

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mctd


cd /data/user_data/mbronars/packages/mctd

CUDA_VISIBLE_DEVICES=0 python main.py +name=my_run

echo "Training completed at $(date)"
