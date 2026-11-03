# 强化学习算法实践

从表格型方法到深度强化学习的实现与练习，代码基于 PyTorch + Gym。

## 目录与算法索引

### 多臂老虎机（探索与利用）

| 文件 | 内容 |
| --- | --- |
| `MAB.py` | 伯努利多臂老虎机，ε-greedy / 上置信界等探索策略 |

### 动态规划（有模型）

| 文件 | 内容 |
| --- | --- |
| `动态规划/PolicyIteration.py` | 策略迭代：策略评估 + 策略改进交替进行 |
| `动态规划/ValueIteration.py` | 价值迭代：直接对最优价值函数做贝尔曼最优备份 |
| `动态规划/Cliff_Walking.py` | 悬崖漫步环境下的 DP 求解 |

### 时序差分（无模型）

| 文件 | 内容 |
| --- | --- |
| `时序差分/Sarsa.py` | 同策略 TD 控制，按实际执行的下一动作更新 |
| `时序差分/Q-leaning.py` | 异策略 TD 控制，按最大 Q 值更新（Q-learning） |
| `时序差分/Dyna-Q.py` | 引入环境模型与规划步骤，提升样本效率 |
| `时序差分/Cliff_Walking.py` | 悬崖漫步环境下的 TD 对比实验 |

### 策略梯度与 Actor-Critic

| 文件 | 内容 |
| --- | --- |
| `策略梯度/PolicyGrad.py` | REINFORCE，蒙特卡洛回报估计梯度 |
| `AC/Actor_Critic.py` | Actor-Critic，用价值函数做基线降低方差 |
| `DDPG.py` | 深度确定性策略梯度，面向连续动作空间 |

### 值函数近似

| 文件 | 内容 |
| --- | --- |
| `DQN/DQN.py` | DQN：经验回放 + 目标网络 |
| `DQN/Double DQN.py` | Double DQN：解耦动作选择与价值评估，缓解高估 |
| `DQN/rl_utils.py` | 训练与绘图的公共工具函数 |

### 基准算法集合

`基准算法/` 下为各主流算法的独立实现，便于横向对比：

| 算法 | 离散动作 | 连续动作 |
| --- | --- | --- |
| Actor-Critic | `Actor-Critic算法.py` | — |
| 策略梯度 | `策略梯度算法.py` | — |
| DQN | `DQN算法.py` / `Double-DQN算法.py` / `Dueling-DQN.py` | — |
| Dyna-Q | `Dyna-Q算法.py` | — |
| DDPG | — | `DDPG算法.py` |
| PPO | `PPO算法(离散动作).py` | `PPO算法(连续动作).py` |
| TRPO | `TRPO(离散动作).py` | `TRPO(连续动作).py` |
| SAC | `SAC(离散动作).py` | `SAC算法(连续动作).py` |

## 环境要求

Python 3.9+，依赖见 `requirements.txt`：

```bash
pip install -r requirements.txt
```

| 库 | 用途 |
| --- | --- |
| `torch` | 神经网络与自动求导 |
| `numpy` | 数值计算 |
| `matplotlib` | 训练曲线与策略可视化 |
| `tqdm` | 训练进度显示 |
| `gym` | 强化学习环境接口 |

## 运行方式

各脚本均可独立运行，训练结束后会弹出曲线图：

```bash
cd 基准算法
python PPO算法\(连续动作\).py
```

## 备注

- `1.py`、`2.py` 为早期随手写的小实验（绘图与列表操作），不属于算法实现。
- 各脚本中的超参数（学习率、折扣因子、网络宽度等）集中在文件顶部的配置区，便于调整对比。
