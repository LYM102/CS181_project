#!/bin/bash
#SBATCH -J nfsp_train
#SBATCH -p normal
#SBATCH --cpus-per-task=4
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -t 12:00:00
#SBATCH -D /public/home/liym2024/CS181/CS181_project
#SBATCH -o /public/home/liym2024/CS181/CS181_project/logs/nfsp_train_%j.out
#SBATCH -e /public/home/liym2024/CS181/CS181_project/logs/nfsp_train_%j.err

set -euo pipefail
source ~/.bashrc
conda activate fmd
export PYTHONUNBUFFERED=1
export PYTHONPATH=/public/home/liym2024/CS181/CS181_project:${PYTHONPATH:-}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="train/results/${TIMESTAMP}"
mkdir -p "${RESULT_DIR}"
mkdir -p logs

MODEL_PATH="${RESULT_DIR}/nfsp_model.pt"
NUM_HANDS=${NUM_HANDS:-1000000}

echo "===== NFSP Self-Play Training Start: ${TIMESTAMP} ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Hands: ${NUM_HANDS}"

python -u train/train_nfsp.py "${NUM_HANDS}" "${MODEL_PATH}"

echo "===== Evaluation ====="
python -u main.py --mode evaluate --agent0 nfsp --agent1 expert --num_hands 3000 --nfsp_model0 "${MODEL_PATH}"
python -u main.py --mode evaluate --agent0 nfsp --agent1 random --num_hands 3000 --nfsp_model0 "${MODEL_PATH}"

echo "===== Training Completed ====="
echo "Model saved to: ${MODEL_PATH}"
