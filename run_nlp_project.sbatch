#!/bin/bash
#
#SBATCH --job-name=nlp-complete-30h
#SBATCH --output=nlp-complete-%j.out
#SBATCH --error=nlp-complete-%j.err
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=30:00:00
#
#SBATCH --mail-user=ysuryavanshi@scu.edu
#SBATCH --mail-type=END,FAIL

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Configuration: COMPLETE COMPREHENSIVE ANALYSIS"
echo "Resource allocation: 2 CPUs, 8GB memory total"
echo "Training: ALL 5 MODELS (BERT, RoBERTa, DeBERTa, HateBERT, ELECTRA)"
echo "Settings: 10 epochs each, batch size 128, full 512 token length"
echo "Time allocation: 30 hours (estimated completion: ~23 hours)"

# Load required modules
echo "Loading TensorFlow module..."
module load TensorFlow/20231212

# Verify GPU availability
echo "GPU Information:"
nvidia-smi

# Set Python path and environment
export PYTHONPATH="${SLURM_SUBMIT_DIR}:$PYTHONPATH"

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR}"

# Run comprehensive NLP project
echo "Starting COMPREHENSIVE NLP Toxic Comment Classification..."
echo "This will train all baseline and advanced models with full evaluation"
echo "Models: Logistic Regression, Random Forest, BERT, RoBERTa, DeBERTa, HateBERT, ELECTRA"
echo "Command: python main.py --all"

# Track progress with timestamps
echo "=========================================="
echo "STARTING COMPLETE PIPELINE at $(date)"
echo "=========================================="

python main.py --all

# Print completion information
echo "=========================================="
echo "COMPLETE PIPELINE FINISHED at $(date)"
echo "=========================================="
echo "Exit status: $?"
echo ""
echo "📁 RESULTS GENERATED:"
echo "   - Baseline models: Logistic Regression, Random Forest"  
echo "   - Advanced models: BERT, RoBERTa, DeBERTa, HateBERT, ELECTRA"
echo "   - Comprehensive comparison and analysis"
echo "   - All plots and visualizations"
echo "   - Per-category performance breakdown"
echo ""
echo "📊 Check results in:"
echo "   - results/ directory for all CSV files"
echo "   - plots/ directory for all visualizations"
echo "   - models/ directory for saved model checkpoints" 