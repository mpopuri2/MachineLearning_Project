import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingQNet(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_dims=(256,256)):
        super().__init__()
        c, t, f = input_shape  # channels, time, features
        in_dim = t * f * c
        self.fc1 = nn.Linear(in_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.val = nn.Linear(hidden_dims[1], 1)
        self.adv = nn.Linear(hidden_dims[1], n_actions)

    def forward(self, x):  # x: [B, C, T, F]
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.val(x)
        a = self.adv(x)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q

class QNet(nn.Module):
    def __init__(self, input_shape, n_actions, hidden_dims=(256,256)):
        super().__init__()
        c, t, f = input_shape
        in_dim = t * f * c
        self.fc1 = nn.Linear(in_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.out = nn.Linear(hidden_dims[1], n_actions)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)