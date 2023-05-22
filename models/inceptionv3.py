import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def inceptionv3(num_classes=10, pretrained=1, **kwargs):
    net = models.inception_v3(pretrained=pretrained)
    net.aux_logits = False

    if pretrained:
        for param in net.parameters():
            param.requires_grad = False

    net.fc = nn.Sequential(
        nn.Linear(net.fc.in_features, 4096),
        nn.Linear(4096, num_classes)
    )

    return net