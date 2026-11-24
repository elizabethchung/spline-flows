#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o runs/%x-%j.out

module load miniforge
conda activate spline
cd ~/spline_flows

nvidia-smi --query-gpu=name,driver_version --format=csv
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
python eval.py
