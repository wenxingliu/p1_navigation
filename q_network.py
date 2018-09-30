import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, seed, fc_units):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = nn.Linear(fc_units, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))        
        return x