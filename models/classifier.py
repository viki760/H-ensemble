import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, backbone, hidden_dim, num_classes, return_feat=False):
        super(Classifier, self).__init__()

        self.backbone = backbone

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)


        self.forward = self.forward_with_hidden if return_feat else self.forward_without_hidden
    
    def forward_with_hidden_logits(self, x):
        x = self.backbone(x)
        x = self.bn(x)
        hidden = x
        x = self.fc(x)
        return x, hidden

    def forward_with_hidden(self, x):
        x = self.backbone(x)
        x = self.bn(x)
        return x
        # hidden = x
        # x = self.fc(x)
        # return x, hidden
    
    def forward_without_hidden(self, x):
        x = self.backbone(x)
        x = self.bn(x)
        x = self.fc(x)
        return x
