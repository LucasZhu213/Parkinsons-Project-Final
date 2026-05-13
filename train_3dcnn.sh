#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --chdir=/csl/users/2026lzhu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=30-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --output=/csl/users/2026lzhu/fastsurfer_%j.out
#SBATCH --error=/csl/users/2026lzhu/fastsurfer_%j.err

export OMP_NUM_THREADS=8

pixi init 3dcnn_workspace
cp pretrained3DCNN.py 3dcnn_workspace
cd gcn_workspace
pixi add "python==3.13"
pixi add --pypi "keras"
pixi add --pypi "torch"
pixi add --pypi "pandas"
pixi add --pypi "scikit-learn"
pixi add --pypi "pandas"
pixi add --pypi "scikit-image"
pixi add --pypi "matplotlib"
pixi add --pypi "classification_models_3D"
echo "HELLO"
pixi run python pretrained3DCNN.py