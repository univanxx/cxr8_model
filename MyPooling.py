import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple


class LogSumExpPool(nn.Module):

    def __init__(self, r = 5):
        super(LogSumExpPool, self).__init__()
        self.r = r

    def forward(self, x):
        
        max_vals = torch.amax(torch.abs(x), dim = (2,3))
        x = max_vals + (1/self.r)*torch.log((1/x.shape[-1])*torch.sum(torch.exp(self.r*(x - max_vals[:,:,None,None])), dim=(2,3)))
        return x