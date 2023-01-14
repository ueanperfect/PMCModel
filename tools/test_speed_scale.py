from PMCModel import Resnet, VGG, VisionTransformer, SwinTransformer, NormalHead,BaseClassifier
from torch.nn.functional import cross_entropy
import torch
from thop import profile
models_dict = {
    'Resnet': ['resnet50', 'resnet34', 'resnet18', 'resnet101'],
    'VGG': ['vgg11', 'vgg13', 'vgg16', 'vgg19'],
    'VisionTransformer': ['vit_b_16', 'vit_b_32', 'vit_l_32'],
    'SwinTransformer': ['swin_t', 'swin_b', 'swin_s']
}

input = torch.randn(1, 3, 224, 224)
result = {'flops':{},'params':{}}

for Model_name in models_dict:
    for model_name in models_dict[Model_name]:
        head = NormalHead(3)
        backbone = eval(Model_name)(model_name=model_name)
        classifier = BaseClassifier(model_name, backbone, head, cross_entropy)
        flops, params = profile(classifier, inputs=(input,))
        result['flops'][model_name] = flops
        result['params'][model_name] = params

import json
with open('data.json', 'w') as fp:
    json.dump(result, fp)


