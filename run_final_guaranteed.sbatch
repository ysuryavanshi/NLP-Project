#!/bin/bash
#SBATCH --job-name=nlp-final-guaranteed
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=nlp-final-guaranteed-%j.out
#SBATCH --error=nlp-final-guaranteed-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ysuryavanshi@scu.edu

echo "========================================================" 
echo "GUARANTEED FINAL RUN & AGGREGATION"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "This script runs the final models with definitive fixes."
echo "========================================================"

# Change to job directory
cd "${SLURM_SUBMIT_DIR}"

# Load required modules
echo "Loading TensorFlow/20231212 module..."
module load TensorFlow/20231212

# Prioritize user's local packages
echo "Setting environment to prioritize local packages..."
export PYTHONPATH="${HOME}/.local/lib/python3.9/site-packages:${PYTHONPATH}"

# Upgrade key packages
echo "Ensuring latest compatible packages are installed..."
pip install --user --upgrade typing-extensions>=4.10.0
pip install --user --upgrade transformers>=4.35.0

# Display GPU information
echo "GPU Information:"
nvidia-smi

echo "Starting Final Aggregation script with guaranteed fixes..."
echo "Command: python run_and_aggregate.py"

# Run the final script which uses the corrected logic
python run_and_aggregate.py

echo "Exit status: $?"

echo "========================================================" 
echo "GUARANTEED FINAL RUN FINISHED at $(date)"
echo "========================================================" 