#!/bin/bash

#SBATCH -J sarsa_train
#SBATCH -p normal
#SBATCH --cpus-per-task=4
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -t 12:00:00
#SBATCH -D /public/home/liym2024/CS181/CS181_project
#SBATCH -o /public/home/liym2024/CS181/CS181_project/logs/sarsa_train_%j.out
#SBATCH -e /public/home/liym2024/CS181/CS181_project/logs/sarsa_train_%j.err

set -euo pipefail

source ~/.bashrc
conda activate fmd

export PYTHONUNBUFFERED=1
export PYTHONPATH=/public/home/liym2024/CS181/CS181_project:${PYTHONPATH:-}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="train/results/${TIMESTAMP}"
mkdir -p "${RESULT_DIR}"
mkdir -p logs

MODEL_PATH="${RESULT_DIR}/sarsa_vs_expert.pkl"
NUM_HANDS=${NUM_HANDS:-200000}

echo "===== SARSA Training (vs Expert) Start: ${TIMESTAMP} ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Hands: ${NUM_HANDS}"
echo "Output model: ${MODEL_PATH}"

python -u train/train_sarsa.py "${NUM_HANDS}" "${MODEL_PATH}"

echo ""
echo "===== Evaluation ====="

# Naive vs Expert
python -u main.py --mode evaluate --agent0 sarsa --agent1 expert --num_hands 3000

# Trained vs Expert
python -u main.py --mode evaluate --agent0 sarsa --agent1 expert --num_hands 3000 --sarsa_model0 "${MODEL_PATH}"

# Trained vs Random (额外参考)
python -u main.py --mode evaluate --agent0 sarsa --agent1 random --num_hands 3000 --sarsa_model0 "${MODEL_PATH}"

echo ""
echo "===== Training Completed ====="
echo "Model saved to: ${MODEL_PATH}"
