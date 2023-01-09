import torch.nn as nn


class VisionTransformer(nn.Module):
    def __init__(self,
                 model_name='vit_b_16'):
        super().__init__()
        self.model_name = model_name
        self.model_names_list = ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14']
        if model_name not in self.model_names_list:
            raise NotImplementedError('we cant impletment this model')
        if model_name == 'vit_b_16':
            from torchvision.models import vit_b_16
            self.model = vit_b_16(pretrained=True)
        elif model_name == 'vit_b_32':
            from torchvision.models import vit_b_32
            self.model = vit_b_32(pretrained=True)
        elif model_name == 'vit_l_16':
            from torchvision.models import vit_l_16
            self.model = vit_l_16(pretrained=True)
        elif model_name == 'vit_l_32':
            from torchvision.models import vit_l_32
            self.model = vit_l_32(pretrained=True)
        elif model_name == 'vit_h_14':
            from torchvision.models import vit_h_14
            self.model = vit_h_14(pretrained=True)
        else:
            raise NotImplementedError('we cant impletment this model')

    def forward(self, x):
        return self.model(x)
