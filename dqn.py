# dqn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),  # Additional hidden layer
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DQCN(nn.Module):
    def __init__(self, input_length, output_dim):
        super(DQCN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(64 * input_length, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        # Reshape the input: [batch_size, sequence_length] -> [batch_size, 1, sequence_length]
        x = x.unsqueeze(1)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)