import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import pickle
import os

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

    def save(self, path='.\\replay_buffer.txt'):
        with open(path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load(self, path='.\\replay_buffer.txt'):
        with open(path, 'rb') as f:
            self.buffer = pickle.load(f)

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Critic,self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, s, a):
        x = torch.cat([s,a], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DDPG:
    def __init__(self,state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, tau, device):
        self.Actor = Actor(state_dim, hidden_dim, action_dim).to(device)
        self.Critic = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.target_Actor = Actor(state_dim, hidden_dim, action_dim).to(device)
        self.target_Critic = Critic(state_dim, hidden_dim, action_dim).to(device)
        # Initialise the target network with the same weights
        self.target_Actor.load_state_dict(self.Actor.state_dict())
        self.target_Critic.load_state_dict(self.Critic.state_dict())

        self.Actor_optimizer = torch.optim.Adam(self.Actor.parameters(), lr=actor_lr)
        self.Critic_optimizer = torch.optim.Adam(self.Critic.parameters(), lr=critic_lr)
        self.action_dim = action_dim
        self.gamma = gamma  # discount factor
        self.tau = tau  # soft update coefficient
        self.device = device

    def save_models(self, save_dir="./models", tag=""):
        os.makedirs(save_dir, exist_ok=True)

        if tag:
            suffix = f"_{tag}"
        else:
            suffix = ""

        torch.save(self.Actor.state_dict(), os.path.join(save_dir, f"Actor_DDPG{suffix}.pth"))
        torch.save(self.Critic.state_dict(), os.path.join(save_dir, f"Critic_DDPG{suffix}.pth"))
        torch.save(self.target_Actor.state_dict(), os.path.join(save_dir, f"target_Actor_DDPG{suffix}.pth"))
        torch.save(self.target_Critic.state_dict(), os.path.join(save_dir, f"target_Critic_DDPG{suffix}.pth"))

    def load_models(self):
        self.Actor.load_state_dict(torch.load('Actor_DDPG.pth'))
        self.Critic.load_state_dict(torch.load('Critic_DDPG.pth'))
        self.target_Actor.load_state_dict(torch.load('target_Actor_DDPG.pth'))
        self.target_Critic.load_state_dict(torch.load('target_Critic_DDPG.pth'))

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.Actor(state).item()

        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        next_q_values = self.target_Critic(next_states, self.target_Actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(self.Critic(states, actions), q_targets))
        self.Critic_optimizer.zero_grad()
        critic_loss.backward()
        self.Critic_optimizer.step()

        actor_loss = - torch.mean(self.Critic(states, self.Actor(states)))
        self.Actor_optimizer.zero_grad()
        actor_loss.backward()
        self.Actor_optimizer.step()

        self.soft_update(self.Actor, self.target_Actor)
        self.soft_update(self.Critic, self.target_Critic)

        return {"critic_loss": float(critic_loss.item()),"actor_loss": float(actor_loss.item())}
