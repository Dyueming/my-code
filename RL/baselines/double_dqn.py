import random
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def dis_to_con(discrete_action, env, action_dim):  # map a discrete action back to continuous
    action_lowbound = env.min_torque  # min continuous action
    action_upbound = env.max_torque  # max continuous action
    return action_lowbound + (discrete_action /(action_dim - 1)) * (action_upbound -action_lowbound)


class PendulumEnv:
    def __init__(self):
        # Gear parameters
        self.max_torque = 2.0  # maximum torque
        self.min_torque = -2.0  # minimum torque
        self.dt = 0.05  # time step
        self.g = 9.81  # gravitational acceleration
        self.length = 1.0  # pole length
        self.m = 0.1  # pole mass
        self.action_dim = 11  # 11 bins for the discrete action
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
    ''' Replay buffer '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # FIFO queue

    def add(self, state, action, reward, next_state, done):  # add a transition to the buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # sample batch_size transitions from the buffer
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # number of transitions currently in the buffer
        return len(self.buffer)

class Qnet(torch.nn.Module):
    ''' Q network with one hidden layer '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # ReLU in the hidden layer
        return self.fc2(x)

class DQN:
    ''' DQN and Double DQN '''
    def __init__(self,
                 state_dim,
                 hidden_dim,
                 action_dim,
                 learning_rate,
                 gamma,
                 epsilon,
                 target_update,
                 device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def max_q_value(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        return self.q_net(state).max().item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q values
        # Max Q value of the next state
        max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
        max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD target
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # MSE loss
        self.optimizer.zero_grad()  # PyTorch accumulates gradients, so zero them explicitly
        dqn_loss.backward()  # backpropagate and update
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # update the target network
        self.count += 1


def train_DQN(agent, env, num_episodes, replay_buffer, minimal_size,batch_size):
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10),
                  desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(state) * 0.005 + max_q_value * 0.995  # smoothing
                    max_q_value_list.append(max_q_value)  # max Q per state
                    action_continuous = dis_to_con(action, env,
                                                   agent.action_dim)
                    next_state, reward, done, _ = env.step([action_continuous])
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(
                            batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list, max_q_value_list



lr = 1e-2
num_episodes = 200
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 50
buffer_size = 5000
minimal_size = 1000
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
env = PendulumEnv()
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.state.shape[0]

action_dim = env.action_dim  # action dimension
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,target_update, device)


return_list, max_q_value_list = train_DQN(agent, env, num_episodes,replay_buffer, minimal_size,batch_size)

episodes_list = list(range(len(return_list)))
mv_return =moving_average(return_list, 5)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Double DQN on {}'.format("PendulumEnv"))
plt.show()

frames_list = list(range(len(max_q_value_list)))
print(len(frames_list))
plt.plot(frames_list, max_q_value_list)
plt.axhline(0, c='orange', ls='--')
plt.axhline(10, c='red', ls='--')
plt.xlabel('Frames')
plt.ylabel('Q value')
plt.title('Double DQN on {}'.format("PendulumEnv"))
plt.show()