from torch import  nn
import torch 


class Discriminator(nn.Module):
     def __init__(self,input_dim):
         super(Discriminator, self).__init__()
         self.fc1 = nn.Linear(input_dim, 64)
         self.fc2 = nn.Linear(64, 1)

     def forward(self, x):
         x = self.fc1(x)
         x=nn.functional.relu(x)
         x = torch.sigmoid(self.fc2(x))
         return x
