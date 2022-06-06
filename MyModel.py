import torch
import torch.nn as nn
import torchvision.models as models
from MyPooling import LogSumExpPool


class MyResNet(nn.Module):
    
    def __init__(self, r=5):
        super(MyResNet, self).__init__()
        
        self.pretrained = models.resnet50(pretrained=True)
        self.pretrained.fc = nn.Identity()
        self.pretrained.avgpool = nn.Identity()
        self.transition = nn.Conv2d(2048, 2048, kernel_size=(2, 2), padding=16, bias=False)
        self.pooling = LogSumExpPool(r=r)
        self.prediction = nn.Linear(in_features=2048, out_features=8, bias=True)
    
    def forward(self, x):
        x = self.pretrained(x)
        x = self.transition(x[:,:,None,None])
        x = self.pooling(x)
        x = self.prediction(torch.flatten(x, start_dim=1))
        return x