from torch import  nn
import torch

class Generator(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x