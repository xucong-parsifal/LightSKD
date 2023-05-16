import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,num_classes=100):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_classes,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 100),
            nn.Sigmoid()
            )
    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.model(x)
        return x
