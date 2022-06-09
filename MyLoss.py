import torch
from torch.nn.modules.loss import BCEWithLogitsLoss


class W_CEL(BCEWithLogitsLoss):
    
    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()
        self.beta_pos = 1
        self.beta_neg = 1
        
    def forward(self, logits, y):
        probs = torch.sigmoid(logits)
        return -(self.beta_pos*torch.log(probs[y == 1]).sum() + self.beta_neg*torch.log((1 - probs)[y == 0]).sum())
        
        