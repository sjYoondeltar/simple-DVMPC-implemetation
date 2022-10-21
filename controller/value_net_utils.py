import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()

        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, states, actions):
        
        x1 = torch.cat([states, actions], dim=1)

        x1 = F.gelu(self.fc1(x1))

        x1 = F.gelu(self.fc2(x1))

        q_value = self.fc3(x1)

        return q_value


class ValueNet(nn.Module):
    def __init__(self, state_size, hidden_size):
        super().__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, states):
        
        x1 = F.leaky_relu(self.fc1(states))
        
        x1 = F.leaky_relu(self.fc2(x1))

        q_value = self.fc3(x1)

        return q_value


class EnsembleValueNet(nn.Module):
    def __init__(self, state_size, hidden_size, num_ensemble):
        super().__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_ensemble)

    def forward(self, states):
        
        x1 = F.leaky_relu(self.fc1(states))
        
        x1 = F.leaky_relu(self.fc2(x1))

        q_value = self.fc3(x1)

        return q_value


class ExperienceReplayMemory:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.buffer_size:
            self.memory.pop(0)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

