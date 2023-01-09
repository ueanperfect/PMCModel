import torch.nn as nn


class SwinTransformer(nn.Module):
    def __init__(self,
                 model_name='swin_t'):
        super().__init__()
        self.model_name = model_name
        self.model_names_list = ['swin_t', 'swin_b', 'swin_s']
        if model_name not in self.model_names_list:
            raise NotImplementedError('we cant impletment this model')
        if model_name == 'swin_t':
            from torchvision.models import swin_t
            self.model = swin_t(pretrained=True)
        elif model_name == 'swin_b':
            from torchvision.models import swin_b
            self.model = swin_b(pretrained=True)
        elif model_name == 'swin_s':
            from torchvision.models import swin_s
            self.model = swin_s(pretrained=True)
        else:
            raise NotImplementedError('we cant impletment this model')

    def forward(self, x):
        return self.model(x)
