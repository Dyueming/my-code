import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import time
from temporal_difference import Cliff_Walking

class DynaQ:
    """ Dyna-Q algorithm """
    def __init__(self,ncol,nrow,epsilon,alpha,gamma,n_planning,n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])  # initialise the Q(s, a) table
        self.n_action = n_action  # number of actions
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # epsilon-greedy parameter
        self.n_planning = n_planning  # Q-planning steps per Q-learning update
        self.model = dict()  # environment model

    def take_action(self, state):  # choose the next action
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def q_learning(self, s0, a0, r, s1):
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

    def update(self, s0, a0, r, s1):
        self.q_learning(s0, a0, r, s1)
        self.model[(s0, a0)] = r, s1  # add to the model
        for _ in range(self.n_planning):  # Q-planning loop
            # Pick a previously seen state-action pair at random
            (s, a), (r, s_) = random.choice(list(self.model.items()))
            self.q_learning(s, a, r, s_)

def DynaQ_CliffWalking(n_planning):
    ncol = 12
    nrow = 4
    env = Cliff_Walking.CliffWalkingEnv(ncol, nrow)
    epsilon = 0.01
    alpha = 0.1
    gamma = 0.9
    agent = DynaQ(ncol, nrow, epsilon, alpha, gamma, n_planning)
    num_episodes = 300  # episodes to run

    return_list = []  # per-episode returns
    for i in range(10):  # 10 progress bars
        # tqdm progress bar
        with tqdm(total=int(num_episodes / 10),
                  desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):  # episodes per progress bar
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done = env.step(action)
                    episode_return += reward  # no discounting in this return
                    agent.update(state, action, reward, next_state)
                    state = next_state
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:  # print the mean return of every 10 episodes
                    pbar.set_postfix({
                        'episode':'%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':'%.3f' % np.mean(return_list[-10:])
                                      })
                pbar.update(1)
    return return_list

np.random.seed(0)
random.seed(0)
n_planning_list = [0, 2, 20]
for n_planning in n_planning_list:
    print('Q-planning steps: %d' % n_planning)
    time.sleep(0.5)
    return_list = DynaQ_CliffWalking(n_planning)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list,return_list,label=str(n_planning) + ' planning steps')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Dyna-Q on {}'.format('Cliff Walking'))
plt.show()