# Texas Hold'em AI — 多人策略训练与评估平台

双人（heads-up）限注德州扑克 AI 智能体训练与评估平台。标准 52 张牌组，固定下注级别。

---

## 1 项目结构

```
CS181_project/
├── main.py                          # 入口：3 种运行模式 (interactive/evaluate/step)
├── requirements.txt                 # Python 依赖
├── cfr_policy.pkl                   # 预训练 CFR 策略缓存
│
├── game/                            # ====== 核心游戏模块 ======
│   ├── constants.py                 # 游戏常量（52 张牌、盲注、下注级别、动作）
│   ├── card.py                      # 牌组 + treys 卡牌工具
│   ├── evaluator.py                 # 手牌评估、比较、胜率计算
│   ├── engine.py                    # 游戏引擎：发牌 → 下注轮 → 摊牌
│   └── cfr_solver.py               # External Sampling MCCFR 求解器
│
├── agents/                          # ====== 智能体实现 ======
│   ├── base_agent.py                # 抽象基类（统一接口）
│   ├── random_agent.py              # 随机策略基线
│   ├── expert_agent.py              # CFR Nash 均衡 Expert
│   ├── sarsa_agent.py               # SARSA on-policy TD 学习
│   ├── nn_mc_agent.py               # BNN-Policy + NN_MC Agent（核心）
│   ├── nfsp_agent.py                # 神经虚拟自博弈（双 DNN）
│   └── aggressive_agent.py          # 激进/紧弱风格 Agent（训练多样性用）
│
├── train/                           # ====== 训练脚本 ======
│   ├── compare_all.py               # 三方对比：BNN vs SARSA vs Expert
│   ├── train_bnn_policy.py          # BNN-Policy 四阶段训练
│   ├── train_expert_distill.py      # Expert 策略蒸馏
│   ├── run_phase3_phase4.py         # 在线 RL + Self-play 训练
│   └── results/policy/              # 训练产出
│       ├── expert_distill_sp_sp.pt  # ★ 最优 BNN 模型
│       └── sarsa_trained.pkl        # ★ 训练好的 SARSA Q-table
│
└── logs/
    └── compare_all.out              # ★ 最终三方对比结果
```

---

## 2 游戏规则

| 参数 | 值 | 说明 |
|------|-----|------|
| 模式 | 2 人 heads-up | 限注德州扑克 |
| 牌组 | **52 张** | 4 花色 × 13 点数 |
| 起始筹码 | 1000 | 每人 |
| 盲注 | SB=5, BB=10 | heads-up: dealer = 小盲 |
| 下注级别 | {10, 20, 40, 80, 160, 320} | 6 级 |
| 最大加注 | **每轮 4 次** | |

动作空间：\(\mathcal{A} = \{\text{Fold}, \text{Call}, \text{Raise}\}\)

---

## 3 智能体总览

| Agent | 核心方法 | 技术水平 |
|-------|---------|---------|
| **Random** | 合法动作均匀随机 | 基线 |
| **Expert** | External Sampling MCCFR → Nash 均衡策略 | 强基线 |
| **SARSA** | Q-table + on-policy TD learning（6720 维状态） | ≈ Expert |
| **NN_MC** | BNN (MC Dropout) + Q-table 混合 | 传统方法 |
| **NFSP** | 双 DNN 自博弈（DQN + 平均策略网络） | 端到端 Nash |
| **BNN-Policy** | BNN 策略网络 + 四阶段训练管线 | ★ 当前最优 |

---

## 4 BNN-Policy 架构（核心）

### 4.1 网络结构

```
输入层 (53 维特征)
    │
    ├─ ResidualBlock(53 → 256)
    │   ├─ Linear + LayerNorm + ReLU
    │   ├─ Dropout(0.15)
    │   ├─ Linear(256→256) + LayerNorm
    │   └─ + 残差连接
    │
    ├─ ResidualBlock(256 → 128)   同上结构
    ├─ ResidualBlock(128 → 64)    同上结构
    │
    └─ Linear(64 → 3) → [Fold, Call, Raise] logits
```

### 4.2 53 维输入特征

| 维度 | 内容 | 说明 |
|------|------|------|
| [0] | 自身胜率 equity | [0,1] |
| [1:5] | 牌面结构（对子/同花/顺子听牌/连通性） | 5 维 |
| [5:9] | 当前轮次 onehot | 4 维 |
| [9:25] | 对手动作历史矩阵（4轮×4 动作频率） | 16 维 |
| [25:41] | 自身动作历史矩阵 | 16 维 |
| [41] | 新公共牌出现标记 | binary |
| [42] | 底池赔率 pot_odds | [0,1] |
| [43] | 筹码底池比 SPR | [0,1] |
| [44:47] | 对手牌力估计（训练真值 / 推理 0.5 掩码） | 3 维 |
| [47:53] | 下注级别/位置/剩余加注/有效筹码/合法动作数 | 6 维 |

### 4.3 MC Dropout（贝叶斯不确定性）

推理时不关闭 Dropout，跑 **20 次随机前向**取均值与方差：

```
20 次采样 → 均值 μ(action) + 方差 σ²(action)
```

- 方差低 → 模型对该决策"有把握"
- 方差高 → 不确定性大 → 倾向保守动作

### 4.4 四阶段训练管线

```
Phase 1: Behavior Cloning  ──→  从 SARSA/Expert 模仿冷启动
Phase 2: DAgger             ──→  SARSA 做在线 oracle，在线纠错
Phase 3: REINFORCE + EWC    ──→  在线 RL 自我提升 + 防遗忘
Phase 4: Self-play          ──→  对抗历史策略快照，提升鲁棒性
```

---

## 5 最终结果

**三方对比**：BNN-SP2 vs SARSA vs Expert，各 3000 手。

| 对局 | 胜率 (WR) | 场均筹码差 |
|------|----------|-----------|
| BNN-SP2 vs Expert | 48.1% vs 48.2% | **BNN +1.25 chips/hand** |
| BNN-SP2 vs SARSA | 83.7% vs 14.9% | **BNN +5.04 chips/hand** |
| SARSA vs Expert | 48.5% vs 47.6% | **SARSA +2.94 chips/hand** |

> 注：AvgR 存在盲注偏移（`chips_before` 在盲注扣除后记录，底池已含 15 筹码盲注），看双方 **差值** 即可衡量真实实力差距。

### 实力排名

```
BNN-SP2  ⪆  SARSA  ⪆  Expert  >>>>  Random
```

三者均达到接近 Nash 均衡的水平，BNN-Policy 在筹码效率上略胜一筹。

### SARSA 改进历程

原始 SARSA 存在关键 bug：最后一动作从未收到终端 reward 更新，导致仅 **9.1%** 胜率 vs Expert。修复后（+ epsilon 衰减 + 训练量翻倍），Q-table 从 117 条增长到 676 条，胜率跃升至 **48.5%**。

---

## 6 快速开始

### 6.1 环境

```bash
pip install -r requirements.txt   # treys + torch + numpy
```

### 6.2 运行评估

```bash
# 三方对比（自动训练对比）
python -u train/compare_all.py

# 任意两两对局
python main.py --mode evaluate --agent0 expert --agent1 sarsa \
    --sarsa_model0 train/results/policy/sarsa_trained.pkl --num_hands 3000
```

### 6.3 可用 Agent 类型

| Key | Agent |
|-----|-------|
| `random` | 随机基线 |
| `expert` | CFR Nash 均衡 |
| `sarsa` | SARSA Q-table |
| `nn_mc` | BNN + Q-table 混合 |
| `nfsp` | 神经虚拟自博弈 |

---

## 7 设计笔记

### 为何选择限注德州扑克

离散下注级别将动作空间约束为 {Fold, Call, Raise}，使得：
- 表格方法（SARSA）在 6720 维状态空间可训练
- CFR 信息集紧凑（~322 个）
- 聚焦策略深度而非下注大小优化

### 状态抽象

手牌通过胜率离散化（20 个 bin）而非原始卡牌值编码，结合轮次、下注级别、底池大小、位置信息，形成 6720 维状态空间。BNN-Policy 使用更丰富的 53 维特征，包含动作历史矩阵、牌面结构、底池赔率等。

### References

- Zinkevich et al. (2008): "Regret Minimization in Games with Incomplete Information"
- Heinrich & Silver (2016): "Deep Reinforcement Learning from Self-Play in Imperfect-Information Games"
- Gal & Ghahramani (2016): "Dropout as a Bayesian Approximation"
- Ross et al. (2011): "A Reduction of Imitation Learning to No-Regret Online Learning" (DAgger)
- Kirkpatrick et al. (2017): "Overcoming catastrophic forgetting in neural networks" (EWC)
