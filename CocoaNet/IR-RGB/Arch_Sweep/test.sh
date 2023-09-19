#!/bin/bash

#SBATCH --job-name=DisNet-IR
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=70G
#SBATCH --cpus-per-task=12  # Request 6 CPU cores

ping google.com