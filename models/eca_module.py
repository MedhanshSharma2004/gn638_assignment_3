import torch
import torch.nn as nn

class ECA_Module(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(output_size = (1, 1))
        self.conv1d = nn.Conv1d(1, 1, kernel_size, padding = 'same')
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        x_pool = self.global_pool(x).squeeze(-1).permute(0, 2, 1)
        y = self.conv1d(x_pool)
        y = self.sigmoid(y)
        x_weighted = x * y.unsqueeze(-1).permute(0, 2, 1, 3)
        return x_weighted