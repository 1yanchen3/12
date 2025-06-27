import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np

class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(192, 64, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, 1, 1, 0)
        )

    def forward(self, t, y):
        return self.net(y)


class ODEblock(nn.Module):
    def __init__(self,odefunc):
        super(ODEblock, self).__init__()
        self.odefunc = odefunc
    def forward(self, x):
        t = torch.from_numpy(np.linspace(0, 1, 5)).cuda()
        out = odeint(self.odefunc, x, t, method='rk4')
        return out[-1]


class ODENet(nn.Module):
    def __init__(self,):
        super(ODENet, self).__init__()
        odefunc = ODEFunc()
        self.odeblock = ODEblock(odefunc)

    def forward(self, x):
        out = self.odeblock(x)
        return out


