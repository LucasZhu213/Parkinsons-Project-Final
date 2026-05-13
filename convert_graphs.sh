#!/bin/bash
#SBATCH --chdir=/csl/users/2026lzhu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=30-00:00:00
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/csl/users/2026lzhu/fastsurfer_%j.out
#SBATCH --error=/csl/users/2026lzhu/fastsurfer_%j.err


pixi init gcn_workspace
cp convert_scan_to_graph.py gcn_workspace
cd gcn_workspace
pixi add "python==3.13"
pixi add --pypi "nibabel"
pixi add --pypi "pandas"
pixi add --pypi "matplotlib"
pixi add --pypi "torch"
pixi add --pypi "torch-geometric"
pixi add --pypi "scipy"

echo "HELLO"
pixi run python convert_scan_to_graph.py