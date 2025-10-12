import torch.nn as nn

class ProjectionMLP(nn.Module):

    def __init__(self, in_dim=100, hid_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),     # Linear layer
            nn.ReLU(inplace=True),          # Relu layer
            nn.Linear(hid_dim, out_dim)     # Linear layer
        )
    def forward(self, x):
        return self.net(x)

