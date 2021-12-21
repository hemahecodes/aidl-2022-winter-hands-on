import torch
import torch.nn as nn


class RegressionModel(nn.Module):
    # You should build your model with at least 2 layers using tanh activation in between
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.h1 = nn.Linear(input_size,hidden_size)
        self.a1 = nn.Tanh()
        self.h2 = nn.Linear(hidden_size,1)
        self.a2 = nn.Tanh()

    def forward(self, x):
        h1 = self.a1(self.h1(x))
        y = self.a2(self.h2(h1))
        return y
        