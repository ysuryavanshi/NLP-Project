#!/bin/bash
#SBATCH --job-name=nlp-final-run
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=nlp-final-run-%j.out
#SBATCH --error=nlp-final-run-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ysuryavanshi@scu.edu

echo "===========================================" 
echo "FINAL RUN & AGGREGATION"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Models to run: DeBERTa, ELECTRA"
echo "This script will train the final models and then aggregate all results."
echo "==========================================="

# Change to job directory
cd "${SLURM_SUBMIT_DIR}"

# Load required modules
echo "Loading TensorFlow/20231212 module..."
module load TensorFlow/20231212

# Prioritize user's local packages to ensure correct versions are used
echo "Setting environment to prioritize local packages..."
export PYTHONPATH="${HOME}/.local/lib/python3.9/site-packages:${PYTHONPATH}"

# Install/upgrade key packages to ensure compatibility
pip install --user --upgrade typing-extensions>=4.10.0
pip install --user --upgrade transformers>=4.35.0

# Display GPU information
echo "GPU Information:"
nvidia-smi

echo "Starting Final Run & Aggregation script..."
echo "Command: python run_and_aggregate.py"

# Run the final script
python run_and_aggregate.py

echo "Exit status: $?"

echo "===========================================" 
echo "FINAL RUN & AGGREGATION FINISHED at $(date)"
echo "===========================================" 