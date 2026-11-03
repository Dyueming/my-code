import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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



class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 device):
        self.policy_net = PolicyNet(state_dim, hidden_dim,
                                    action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate)  # Adam optimiser
        self.gamma = gamma  # discount factor
        self.device = device

    def take_action(self, state):  # sample from the action distribution
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # start from the last step
            reward = reward_list[i]
            state = torch.tensor([state_list[i]],
                                 dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G  # per-step loss
            loss.backward()  # backpropagate
        self.optimizer.step()  # gradient descent


learning_rate = 1e-3
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")


env = CartPoleEnv()
torch.manual_seed(0)
state_dim = env.state.shape[0]
action_dim = env.action_force_mag.shape[0]  # action dimension
agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma,device)

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            transition_dict = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []
            }
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
            agent.update(transition_dict)
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
plt.title('REINFORCE on {}'.format("CartPoleEnv"))
plt.show()

mv_return = moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('REINFORCE on {}'.format("CartPoleEnv"))
plt.show()