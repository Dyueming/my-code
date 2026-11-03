# Personal Code Repository

A collection of personal projects and code practice.

## Projects

| Directory | Contents | Stack |
| --- | --- | --- |
| [`RL/`](RL/) | Reinforcement learning implementations: bandits, dynamic programming, temporal difference, policy gradient, DQN / DDPG / PPO / TRPO / SAC | Python · PyTorch · Gym |
| [`Meta-RL/`](Meta-RL/) | Load-following control for a nuclear power unit: DDPG agent with Optuna reward-weight meta-optimisation, coupled to a RELAP5 thermal-hydraulic simulation | Python · PyTorch · Optuna · RELAP5 |

## Layout

```
.
├── README.md
├── .gitignore
├── .gitattributes
├── RL/                          # Reinforcement learning practice
│   ├── README.md                # Algorithm index and run instructions
│   ├── requirements.txt
│   ├── MAB.py                   # Multi-armed bandit
│   ├── DDPG.py
│   ├── dynamic_programming/     # Policy iteration, value iteration, cliff walking
│   ├── temporal_difference/     # Sarsa, Q-learning, Dyna-Q
│   ├── policy_gradient/         # REINFORCE
│   ├── AC/                      # Actor-Critic
│   ├── DQN/                     # DQN, Double DQN
│   └── baselines/               # 12 reference implementations
└── Meta-RL/                     # Nuclear load-following RL control
    ├── Env.py                   # Environment wrapper around RELAP5
    ├── DDPG.py                  # Agent and replay buffer
    ├── Train-776.py             # 80% power condition
    ├── Train-873.py             # 90% power condition
    ├── Meta_Optimize_776.py     # Optuna reward-weight search (80%)
    └── Meta_Optimize_873.py     # Optuna reward-weight search (90%)
```

New projects go in their own subdirectory, each with its own `README.md`.

## Usage

Clone the repository:

```bash
git clone <repository-url>
```

Day-to-day commits:

```bash
git add .
git commit -m "feat: describe the change"
git push
```

## Commit Convention

| Prefix | Purpose |
| --- | --- |
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Refactoring without behaviour change |
| `chore` | Build, dependencies, configuration |
| `test` | Tests |
