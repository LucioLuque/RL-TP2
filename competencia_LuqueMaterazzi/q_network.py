import os
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, input_dim: int = 1, output_dim: int = 2):
        super().__init__()
        self.q_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.q_net(x)

    def get_action(self, x):
        with torch.no_grad():
            q_values = self.forward(x)
            action = torch.argmax(q_values, dim=-1)
        
        return action.item()

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))