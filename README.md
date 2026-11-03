# 个人代码仓库

存放个人项目与代码实践的仓库。

## 项目列表

| 目录 | 内容 | 技术栈 |
| --- | --- | --- |
| [`RL/`](RL/) | 强化学习算法实践：多臂老虎机、动态规划、时序差分、策略梯度、DQN / DDPG / PPO / TRPO / SAC 等 | Python · PyTorch · Gym |

## 目录结构

```
.
├── README.md
├── .gitignore
├── .gitattributes
└── RL/                  # 强化学习算法实践
    ├── README.md        # 算法索引与运行说明
    ├── requirements.txt
    ├── MAB.py
    ├── DDPG.py
    ├── 动态规划/
    ├── 时序差分/
    ├── 策略梯度/
    ├── AC/
    ├── DQN/
    └── 基准算法/
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
