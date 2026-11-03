import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # tqdm renders loop progress bars
import time
import Cliff_Walking
# Sarsa agent
class Sarsa:
    """ Sarsa algorithm """
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        """
        Initialise the Q table and learning parameters
        ncol, nrow: grid columns and rows
        epsilon: exploration probability for epsilon-greedy
        alpha: learning rate
        gamma: discount factor
        n_action: number of actions
        """
        self.Q_table = np.zeros([nrow * ncol, n_action])  # rows x cols states, 4 actions
        self.n_action = n_action  # number of available actions
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # epsilon-greedy exploration probability

    def take_action(self, state):
        """
        Choose an action with epsilon-greedy
        state: current state
        Returns: the selected action
        """
        if np.random.random() < self.epsilon:
            # With probability eps pick a random action
            action = np.random.randint(self.n_action)
        else:
            # With probability 1-eps pick the action with the largest Q value
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):
        """
        Best action for a given state, used to print the policy
        state: current state
        Returns: optimal action (ties are all marked as 1)
        """
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1, a1):
        """
        Update the Q value
        s0: current state
        a0: current action
        r: reward
        s1: next state
        a1: next action
        """
        # Compute the temporal-difference (TD) error
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        # Update Q with the TD error
        self.Q_table[s0, a0] += self.alpha * td_error

# Set the grid columns and rows
ncol = 12
nrow = 4
# Create the cliff walking environment
env = Cliff_Walking.CliffWalkingEnv(ncol, nrow)
# Fix the seed for reproducibility
np.random.seed(0)
# Sarsa hyperparameters
epsilon = 0.1
alpha = 0.1
gamma = 0.9
# Create the Sarsa agent
agent = Sarsa(ncol, nrow, epsilon, alpha, gamma)
num_episodes = 500  # total episodes
# Store the per-episode return (reward)
return_list = []
# Show progress in 10 groups of 50 episodes
for i in range(10):
    # tqdm progress bar
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0  # total return of this episode
            state = env.reset()  # reset the environment
            action = agent.take_action(state)  # initial action via epsilon-greedy
            done = False  # episode termination flag
            # Start an episode
            while not done:
                next_state, reward, done = env.step(action)  # take the action
                next_action = agent.take_action(next_state)  # action for the next state
                episode_return += reward  # accumulate the step reward
                # Update the Q table
                agent.update(state, action, reward, next_state, next_action)
                # Update the current state and action
                state = next_state
                action = next_action
            # Store the total return of this episode
            return_list.append(episode_return)
            # Print the mean return every 10 episodes
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return': '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)  # advance the progress bar

# Plot the return against episodes
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
print('Sarsa converged to policy:')
print_agent(agent, env, action_meaning, list(range(37, 47)), [47])