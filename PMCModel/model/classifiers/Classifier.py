import torch
import torch.nn as nn


class BaseClassifier(nn.Module):
    def __init__(self,
                 model_name,
                 backbone,
                 head,
                 loss, ):
        super().__init__()
        self.model_name = model_name
        self.backbone = backbone
        self.head = head
        self.loss = loss

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

    def training(self, x, y):
        re = self.forward(x)
        loss = self.loss(re, y)
        return loss

    def predicting(self, x):
        x = self.forward(x)
        return torch.argmax(x, 1)
