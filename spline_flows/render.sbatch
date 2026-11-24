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

export MUJOCO_GL=egl
python render.py
