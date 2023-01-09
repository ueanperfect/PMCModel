import torch.nn as nn
import torch


class NormalHead(nn.Module):
    def __init__(self,
                 classes_number,
                 input_shape=1000,
                 ):
        super().__init__()
        self.input_shape = input_shape
        self.class_number = classes_number
        self.model = nn.Sequential(
            nn.Linear(input_shape, 256, bias=True),
            nn.Linear(256, self.class_number, bias=True))

    def forward(self, x):
        x = self.model(x)
        return x
