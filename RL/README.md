# Reinforcement Learning Practice

Implementations and exercises spanning tabular methods through deep reinforcement learning, built with PyTorch and Gym.

## Algorithm Index

### Multi-armed bandit (exploration vs. exploitation)

| File | Contents |
| --- | --- |
| `MAB.py` | Bernoulli multi-armed bandit with epsilon-greedy and UCB-style exploration |

### Dynamic programming (model-based)

| File | Contents |
| --- | --- |
| `dynamic_programming/PolicyIteration.py` | Policy iteration: alternating policy evaluation and improvement |
| `dynamic_programming/ValueIteration.py` | Value iteration: Bellman optimality backups directly on the value function |
| `dynamic_programming/Cliff_Walking.py` | Dynamic programming solution for the cliff walking environment |

### Temporal difference (model-free)

| File | Contents |
| --- | --- |
| `temporal_difference/Sarsa.py` | On-policy TD control, updates with the action actually taken |
| `temporal_difference/q_learning.py` | Off-policy TD control, updates with the maximum Q value |
| `temporal_difference/Dyna-Q.py` | Adds a learned model and planning steps to improve sample efficiency |
| `temporal_difference/Cliff_Walking.py` | TD comparison on the cliff walking environment |

### Policy gradient and actor-critic

| File | Contents |
| --- | --- |
| `policy_gradient/PolicyGrad.py` | REINFORCE with Monte-Carlo return estimates |
| `AC/Actor_Critic.py` | Actor-Critic using a value function as the baseline |
| `DDPG.py` | Deep deterministic policy gradient for continuous action spaces |

### Value function approximation

| File | Contents |
| --- | --- |
| `DQN/DQN.py` | DQN with experience replay and a target network |
| `DQN/double_dqn.py` | Double DQN: decouples action selection from evaluation to reduce overestimation |
| `DQN/rl_utils.py` | Shared training and plotting helpers |

### Baseline implementations

`baselines/` holds standalone implementations for side-by-side comparison:

| Algorithm | Discrete action | Continuous action |
| --- | --- | --- |
| Actor-Critic | `actor_critic.py` | — |
| Policy gradient | `policy_gradient.py` | — |
| DQN | `dqn.py` / `double_dqn.py` / `dueling_dqn.py` | — |
| Dyna-Q | `dyna_q.py` | — |
| DDPG | — | `ddpg.py` |
| PPO | `ppo_discrete.py` | `ppo_continuous.py` |
| TRPO | `trpo_discrete.py` | `trpo_continuous.py` |
| SAC | `sac_discrete.py` | `sac_continuous.py` |

## Requirements

Python 3.9+, dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

| Package | Purpose |
| --- | --- |
| `torch` | Neural networks and automatic differentiation |
| `numpy` | Numerical computing |
| `matplotlib` | Training curves and policy visualisation |
| `tqdm` | Training progress bars |
| `gym` | Reinforcement learning environment interface |

## Running

Each script runs standalone and opens a training curve when finished:

```bash
cd baselines
python ppo_continuous.py
```

## Notes

- `1.py` and `2.py` are early scratch scripts (a plot and a list exercise), not algorithm implementations.
- Hyperparameters (learning rate, discount factor, hidden width) sit in a configuration block at the top of each script for easy comparison.
