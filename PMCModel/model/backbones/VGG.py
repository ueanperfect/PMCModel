import torch.nn as nn


class VGG(nn.Module):
    def __init__(self,
                 model_name='vgg16'):
        super().__init__()
        self.model_name = model_name
        self.model_names_list = ['vgg11', 'vgg13', 'vgg16', 'vgg19']
        if model_name not in self.model_names_list:
            raise NotImplementedError('we cant impletment this model')
        if model_name == 'vgg11':
            from torchvision.models import vgg11
            self.model = vgg11(pretrained=True)
        elif model_name == 'vgg16':
            from torchvision.models import vgg16
            self.model = vgg16(pretrained=True)
        elif model_name == 'vgg13':
            from torchvision.models import vgg13
            self.model = vgg13(pretrained=True)
        elif model_name == 'vgg19':
            from torchvision.models import vgg19
            self.model = vgg19(pretrained=True)
        else:
            raise NotImplementedError('we cant impletment this model')

    def forward(self, x):
        return self.model(x)
