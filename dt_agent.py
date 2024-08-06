import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import random
from dt import DecisionTransformer
from torch.utils.tensorboard import SummaryWriter

class DTAgent:
    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DecisionTransformer(state_size, action_size).to(self.device)
        self.target_model = DecisionTransformer(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.writer = SummaryWriter()
        self.update_target_every = 10
        self.steps = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = torch.zeros((1, self.action_size)).to(self.device)
        reward = torch.zeros((1, 1)).to(self.device)
        action_pred = self.model(state, action, reward)
        return action_pred.argmax().item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(np.eye(self.action_size)[actions]).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.model(states, actions, rewards)
        next_actions = torch.zeros((batch_size, self.action_size)).to(self.device)
        next_rewards = torch.zeros((batch_size, 1)).to(self.device)
        next_q_values = self.target_model(next_states, next_actions, next_rewards).max(1)[0].detach()
        target_q_values = rewards.squeeze() + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())