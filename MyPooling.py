import torch
import torch.nn as nn


class LogSumExpPool(nn.Module):
    # Inherit from Module class
    def __init__(self, r = 5):
        super(LogSumExpPool, self).__init__()
        self.r = r
    # Log-Sum-Exp Padding realization from https://arxiv.org/abs/1705.02315
    def forward(self, x):
        max_vals = torch.amax(torch.abs(x), dim = (2,3))
        x = max_vals + (1/self.r)*torch.log(1e-7 + (1/x.shape[-1])*torch.sum(torch.exp(self.r*(x - max_vals[:,:,None,None])), dim=(2,3)))
        return x