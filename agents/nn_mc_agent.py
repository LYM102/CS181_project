# agents/nn_mc_agent.py — legacy re-exports

from agents.belief_net import (
    BNNWithMCDropout,
    calibrate_temperature,
    collect_bnn_training_data,
    collect_bnn_data_sarsa_distill,
    train_bnn,
    train_bnn_multitask,
    _equity_to_strength_label,
    _equity_to_strength_label_5class,
)
from agents.belief_features import BeliefFeatureEncoder
from agents.belief_sarsa_agent import BeliefSarsaAgent
from agents.l1_agent import L1Agent, NN_MCAgent
from agents.l2_agent import L2Agent
from agents.l3_agent import (
    L3Agent,
    BNN_PolicyNet,
    BNN_PolicyAgent,
    ResidualBlock,
    collect_expert_policy_data,
    train_bnn_policy_kl,
)

__all__ = [
    "BNNWithMCDropout",
    "BeliefFeatureEncoder",
    "BeliefSarsaAgent",
    "L1Agent",
    "L2Agent",
    "L3Agent",
    "NN_MCAgent",
    "BNN_PolicyNet",
    "BNN_PolicyAgent",
    "calibrate_temperature",
    "collect_bnn_training_data",
    "collect_bnn_data_sarsa_distill",
    "train_bnn",
    "train_bnn_policy_kl",
    "collect_expert_policy_data",
]
