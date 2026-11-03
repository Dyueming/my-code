import random
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import collections


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


class PendulumEnv:
    def __init__(self):
        # Gear parameters
        self.max_torque = 2.0  # maximum torque
        self.min_torque = -2.0  # minimum torque
        self.dt = 0.05  # time step
        self.g = 9.81  # gravitational acceleration
        self.length = 1.0  # pole length
        self.m = 0.1  # pole mass
        self.action_dim = 1  # currently a single continuous action
        self.theta_threshold_radians = 12 * np.pi / 180  # angle threshold, about 12 degrees
        self.x_threshold = 2.4  # horizontal position threshold
        self.reset()

    def reset(self):
        # Random initial pole angle and angular velocity
        self.theta = np.random.uniform(-np.pi/4, np.pi/4)  # pole angle
        self.theta_dot = np.random.uniform(-0.5, 0.5)  # pole angular velocity
        self.state=np.array([self.theta, self.theta_dot])
        return self.state


    def step(self, action):
        # Torque range: the action should stay within [-2, 2]
        torque = np.clip(action, self.min_torque, self.max_torque)

        # Update the state with the physical equations
        g, l, m = self.g, self.length, self.m
        theta, theta_dot = self.theta, self.theta_dot

        # Equations of motion
        theta_acc = (-3 * g / (2 * l) * np.sin(theta + np.pi) + torque / (m * l ** 2))  # angular acceleration
        theta_dot += theta_acc * self.dt  # update the angular velocity
        theta += theta_dot * self.dt  # update the angle

        # Keep theta in range
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        # Update the state
        self.theta = theta
        self.theta_dot = theta_dot

        # Reward function (closer to balance means higher reward)
        reward = - (theta ** 2 + 0.1 * theta_dot ** 2 + 0.001 * torque ** 2)  # negative reward

        # Return next state, reward, done flag and extra info
        done = (theta < -self.theta_threshold_radians or
                theta > self.theta_threshold_radians)
        return self.state, reward, done, {}  # next_state, reward, done, info

    def render(self):
        # Optional
        pass

    def close(self):
        # Clean up resources
        pass

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # largest action the env accepts

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)  # concat state and action
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class DDPG:
    ''' DDPG algorithm '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # Initialise the target critic with the critic weights
        self.target_critic.load_state_dict(self.critic.state_dict())
        # Initialise the target actor with the actor weights
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma  # std of the Gaussian noise (mean fixed at 0)
        self.tau = tau  # soft update coefficient
        self.action_dim = action_dim
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state).item()
        # Add noise to the action for exploration
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)  # soft update the actor
        self.soft_update(self.critic, self.target_critic)  # soft update the critic

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

actor_lr = 3e-4
critic_lr = 3e-3
num_episodes = 200
hidden_dim = 64
gamma = 0.98
tau = 0.005  # soft update coefficient
buffer_size = 10000
minimal_size = 1000
batch_size = 64
sigma = 0.01  # Gaussian noise std
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'Pendulum-v0'
env = PendulumEnv()
random.seed(0)
np.random.seed(0)
state_dim = env.state.shape[0]
action_dim = env.action_dim  # action dimension
torch.manual_seed(0)
replay_buffer =ReplayBuffer(buffer_size)
action_bound = max(abs(env.max_torque), abs(env.min_torque))  # max action
agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)

return_list =train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)


episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format(env_name))
plt.show()

mv_return =moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format(env_name))
plt.show()