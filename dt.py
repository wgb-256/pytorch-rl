import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter

class DecisionTransformer(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128, nhead=4, num_layers=5):
        super(DecisionTransformer, self).__init__()
        self.state_embedding = nn.Linear(state_size, hidden_size)
        self.action_embedding = nn.Linear(action_size, hidden_size)
        self.reward_embedding = nn.Linear(1, hidden_size)
        self.pos_encoding = nn.Parameter(torch.randn(1, 3, hidden_size))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(hidden_size, action_size)

    def forward(self, states, actions, rewards):
        # Embed each component
        states = self.state_embedding(states)  # (batch_size, hidden_size)
        actions = self.action_embedding(actions)  # (batch_size, hidden_size)
        rewards = self.reward_embedding(rewards)  # (batch_size, hidden_size)

        # Stack the embeddings
        x = torch.stack((states, actions, rewards), dim=1)  # (batch_size, 3, hidden_size)

        # Add positional encoding
        x = x + self.pos_encoding

        # Transformer expects (seq_len, batch_size, hidden_size)
        x = x.transpose(0, 1)

        # Pass through transformer
        x = self.transformer(x)

        # Take the last output for prediction
        x = x[-1]

        return self.fc_out(x)