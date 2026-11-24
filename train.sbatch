#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o runs/%x-%j.out

# Usage (from ~/spline_flows):  sbatch -J spline train.sbatch spline
#                               sbatch -J waypoint train.sbatch waypoint

module load miniforge
conda activate spline
cd ~/spline_flows

echo "python: $(which python)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
python train.py --target "$1"
