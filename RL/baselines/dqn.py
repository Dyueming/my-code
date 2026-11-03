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


class CartPoleEnv:
    def __init__(self):
        self.gravity = 9.8  # gravitational acceleration
        self.masscart = 1.0  # cart mass
        self.masspole = 0.1  # pole mass
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5  # pole length
        self.polemass_length = self.masspole * self.length
        self.action_force_mag = np.array([-10.0,
                                   -5.0,
                                   0.0,
                                   5.0,
                                   10.0])  # force magnitude (six actions only)
        self.tau = 0.02  # time step
        self.theta_threshold_radians = 12 * 2 * np.pi / 360  # pole angle threshold
        self.x_threshold = 2.4  # cart position threshold
        self.reset()

    def reset(self):
        self.state = np.array([0.0,  # position
                                0.0,  # velocity
                                np.random.uniform(-0.05, 0.05),  # pole angle
                                0.0])  # pole angular velocity
        return self.state

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state

        # Apply the force
        force=self.action_force_mag[action]

        # Physical equations
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        theta_acc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass))
        x_acc = temp - self.polemass_length * theta_acc * costheta / self.total_mass

        x += self.tau * x_dot
        x_dot += self.tau * x_acc
        theta += self.tau * theta_dot
        theta_dot += self.tau * theta_acc

        # Update the state
        self.state = np.array([x, x_dot, theta, theta_dot])

        # Check for termination
        done = theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians or x < -self.x_threshold or x > self.x_threshold

        # Reward function
        reward = 1.0 if not done else 0.0  # +1 per step, 0 on failure

        return self.state, reward, done, {}

    def render(self):
        # Visualisation is optional
        pass

    def close(self):
        # Add resource cleanup here if needed
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
    ''' DQN algorithm '''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q network
        # Target network
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        # Adam optimiser
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # epsilon-greedy
        self.target_update = target_update  # target network update frequency
        self.count = 0  # update counter
        self.device = device

    def take_action(self, state):  # epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

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
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
            -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )  # TD target
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # MSE loss
        self.optimizer.zero_grad()  # PyTorch accumulates gradients, so zero them explicitly
        dqn_loss.backward()  # backpropagate and update
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # update the target network
        self.count += 1

lr = 2e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
env = CartPoleEnv()
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.state.shape[0]
action_dim = env.action_force_mag.shape[0]  # action dimension
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                # Start Q-network training only after the buffer exceeds a threshold
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
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

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format("CartPoleEnv"))
plt.show()

mv_return = moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format("CartPoleEnv"))
plt.show()