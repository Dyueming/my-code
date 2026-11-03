import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # tqdm是显示循环进度条的库
import time
import Cliff_Walking
# Sarsa 算法类
class Sarsa:
    """ Sarsa算法 """
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        """
        初始化Q表格和学习参数
        ncol, nrow: 网格的列数和行数
        epsilon: ε-贪婪策略中的探索概率
        alpha: 学习率
        gamma: 折扣因子
        n_action: 动作数
        """
        self.Q_table = np.zeros([nrow * ncol, n_action])  # 初始化Q表，状态数=行数×列数，动作数=4
        self.n_action = n_action  # 可选动作数量
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # ε-贪婪策略的探索概率

    def take_action(self, state):
        """
        根据ε-贪婪策略选择一个动作
        state: 当前状态
        返回: 选择的动作
        """
        if np.random.random() < self.epsilon:
            # 以ε的概率随机选择动作
            action = np.random.randint(self.n_action)
        else:
            # 以1-ε的概率选择当前状态下Q值最大的动作
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):
        """
        获取给定状态下的最佳动作，用于打印策略
        state: 当前状态
        返回: 最优动作（若多个动作相同，将其标记为1）
        """
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1, a1):
        """
        更新Q值
        s0: 当前状态
        a0: 当前动作
        r: 奖励
        s1: 下一个状态
        a1: 下一个动作
        """
        # 计算时间差分误差（TD误差）
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        # 根据TD误差更新Q值
        self.Q_table[s0, a0] += self.alpha * td_error

# 设置网格的列数和行数
ncol = 12
nrow = 4
# 创建悬崖漫步环境
env = Cliff_Walking.CliffWalkingEnv(ncol, nrow)
# 固定随机种子以保证结果可重复
np.random.seed(0)
# 设置Sarsa算法的参数
epsilon = 0.1
alpha = 0.1
gamma = 0.9
# 创建Sarsa算法的智能体
agent = Sarsa(ncol, nrow, epsilon, alpha, gamma)
num_episodes = 500  # 运行的总回合数
# 存储每个回合的回报（reward）
return_list = []
# 分10组显示进度，每组50回合
for i in range(10):
    # tqdm用于显示进度条
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0  # 存储单回合的总回报
            state = env.reset()  # 重置环境
            action = agent.take_action(state)  # 根据ε-贪婪策略选择初始动作
            done = False  # 回合结束标志
            # 开始一个回合
            while not done:
                next_state, reward, done = env.step(action)  # 执行动作
                next_action = agent.take_action(next_state)  # 下一个状态选择动作
                episode_return += reward  # 累加每步的回报
                # 更新Q表
                agent.update(state, action, reward, next_state, next_action)
                # 更新当前状态和动作
                state = next_state
                action = next_action
            # 存储当前回合的总回报
            return_list.append(episode_return)
            # 每运行10个回合，显示平均回报
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return': '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)  # 更新进度条

# 绘制回报随回合数变化的图像
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Sarsa on {}'.format('Cliff Walking'))
plt.show()

def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * env.ncol + j) in end:
                print('EEEE', end=' ')
            else:
                a = agent.best_action(i * env.ncol + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()

action_meaning = ['^', 'v', '<', '>']
print('Sarsa算法最终收敛得到的策略为：')
print_agent(agent, env, action_meaning, list(range(37, 47)), [47])