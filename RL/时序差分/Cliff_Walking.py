# 悬崖漫步环境类
class CliffWalkingEnv:
    def __init__(self, ncol, nrow):
        """初始化环境，设定网格的行数和列数"""
        self.nrow = nrow  # 行数
        self.ncol = ncol  # 列数
        self.x = 0  # 智能体的横坐标（初始为最左侧）
        self.y = self.nrow - 1  # 智能体的纵坐标（初始为最底行）

    def step(self, action):
        """
        执行动作并更新位置
        输入: action (0: 上, 1: 下, 2: 左, 3: 右)
        返回: 下一状态, 奖励, 是否结束
        """
        # 定义4种动作：上、下、左、右，对应位置坐标的变化
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        # 更新智能体的位置，确保不超出边界
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        # 将当前位置转换为状态
        next_state = self.y * self.ncol + self.x
        # 每走一步奖励为-1
        reward = -1
        done = False
        # 检查是否掉入悬崖（靠近右下方的区域）或到达终点
        if self.y == self.nrow - 1 and self.x > 0:
            done = True
            # 如果掉入悬崖，给负奖励；否则为终点，结束游戏
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done

    def reset(self):
        """重置智能体位置到起始点"""
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x