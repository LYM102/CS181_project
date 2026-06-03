#!/bin/bash

#SBATCH -J nn_mc_train
#SBATCH -p normal
#SBATCH --cpus-per-task=4
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -t 12:00:00
#SBATCH -D /public/home/liym2024/CS181/CS181_project
#SBATCH -o /public/home/liym2024/CS181/CS181_project/logs/nn_mc_train_%j.out
#SBATCH -e /public/home/liym2024/CS181/CS181_project/logs/nn_mc_train_%j.err

set -euo pipefail

source ~/.bashrc
conda activate fmd

export PYTHONUNBUFFERED=1
export PYTHONPATH=/public/home/liym2024/CS181/CS181_project:${PYTHONPATH:-}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="train/results/${TIMESTAMP}"
mkdir -p "${RESULT_DIR}"
mkdir -p logs

MODEL_PATH="${RESULT_DIR}/nn_mc_vs_expert.pt"
BNN_DATA_HANDS=${BNN_DATA_HANDS:-20000}
NUM_HANDS=${NUM_HANDS:-50000}
SARSA_MODEL=${SARSA_MODEL:-"train/sarsa_final.pkl"}

echo "===== NN_MC Training (BNN + MC Q-table vs Expert) Start: ${TIMESTAMP} ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "BNN data hands: ${BNN_DATA_HANDS}"
echo "MC training hands: ${NUM_HANDS}"
echo "SARSA model: ${SARSA_MODEL}"
echo "Output model: ${MODEL_PATH}"

python -u train/train_nn_mc.py "${NUM_HANDS}" "${MODEL_PATH}" "${BNN_DATA_HANDS}" "${SARSA_MODEL}"

echo ""
echo "===== Evaluation ====="

# Trained NN_MC vs Expert
python -u main.py --mode evaluate --agent0 nn_mc --agent1 expert --num_hands 3000 --nn_mc_model0 "${MODEL_PATH}"

# Trained NN_MC vs Random
python -u main.py --mode evaluate --agent0 nn_mc --agent1 random --num_hands 3000 --nn_mc_model0 "${MODEL_PATH}"

echo ""
echo "===== Training Completed ====="
echo "Model saved to: ${MODEL_PATH}"
