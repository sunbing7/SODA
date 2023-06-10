import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary

def shufflenetv2(num_classes=10, pretrained=1, **kwargs):
    net = models.shufflenet_v2_x1_0(pretrained=pretrained)
    net.aux_logits = False

    if pretrained:
        for param in net.parameters():
            param.requires_grad = False

    net.fc = nn.Sequential(
        nn.Linear(net.fc.in_features, 4096),
        nn.Linear(4096, num_classes)
    )

    return net

def test():
    net = shufflenetv2()

    summary(net, (3, 200, 200))
#test()