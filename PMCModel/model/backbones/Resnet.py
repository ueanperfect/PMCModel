import torch.nn as nn
class Resnet(nn.Module):
    def __init__(self,
                 model_name='resnet50'):
        super().__init__()
        self.model_name = model_name
        self.model_names_list = ['resnet50', 'resnet34', 'resnet18', 'resnet101']
        if model_name not in self.model_names_list:
            raise NotImplementedError('we cant impletment this model')
        if model_name == 'resnet50':
            from torchvision.models import resnet50
            self.model = resnet50(pretrained=True)
        elif model_name == 'resnet34':
            from torchvision.models import resnet34
            self.model = resnet34(pretrained=True)
        elif model_name == 'resnet18':
            from torchvision.models import resnet18
            self.model = resnet18(pretrained=True)
        elif model_name == 'resnet101':
            from torchvision.models import resnet101
            self.model = resnet101(pretrained=True)
        else:
            raise NotImplementedError('we cant impletment this model')

    def forward(self,x):
        return self.model(x)