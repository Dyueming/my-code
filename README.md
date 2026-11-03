# 个人代码仓库

存放个人项目与代码实践的仓库。

## 项目列表

| 目录 | 内容 | 技术栈 |
| --- | --- | --- |
| [`RL/`](RL/) | 强化学习算法实践：多臂老虎机、动态规划、时序差分、策略梯度、DQN / DDPG / PPO / TRPO / SAC 等 | Python · PyTorch · Gym |
| [`Meta-RL/`](Meta-RL/) | 核电机组负荷跟踪控制：DDPG 智能体 + Optuna 奖励权重元优化，耦合 RELAP5 热工水力仿真 | Python · PyTorch · Optuna · RELAP5 |

## 目录结构

```
.
├── README.md
├── .gitignore
├── .gitattributes
├── RL/                  # 强化学习算法实践
│   ├── README.md        # 算法索引与运行说明
│   ├── requirements.txt
│   ├── MAB.py
│   ├── DDPG.py
│   ├── 动态规划/
│   ├── 时序差分/
│   ├── 策略梯度/
│   ├── AC/
│   ├── DQN/
│   └── 基准算法/
└── Meta-RL/             # 核电负荷跟踪强化学习控制
    ├── Env.py           # 环境封装（对接 RELAP5）
    ├── DDPG.py          # 智能体与经验回放
    ├── Train-776.py     # 80% 功率工况训练
    ├── Train-873.py     # 90% 功率工况训练
    ├── Meta_Optimize_776.py  # Optuna 奖励权重搜索（80%）
    └── Meta_Optimize_873.py  # Optuna 奖励权重搜索（90%）
```

新增项目时按子目录划分，每个目录自带 `README.md`：

```
projects/
└── <项目名>/
    ├── README.md
    └── ...
```

## 使用方式

克隆到本地：

```bash
git clone <仓库地址>
```

日常提交：

```bash
git add .
git commit -m "feat: 说明这次改了什么"
git push
```

## 提交信息规范（约定式提交）

| 前缀 | 用途 |
| --- | --- |
| `feat` | 新功能 |
| `fix` | 修复问题 |
| `docs` | 仅文档变更 |
| `refactor` | 重构，不改变行为 |
| `chore` | 构建、依赖、配置等杂项 |
| `test` | 测试相关 |
