from torchvision.models import alexnet as alexnet_tv
import torch.nn as nn


def alexnet(num_classes=10, pretrained=1):
    net = alexnet_tv(pretrained=pretrained)
    net.classifier[6] = nn.Linear(4096, num_classes)    # nn.Linear in pytorch is a fully connected layer
                                                        # The convolutional layer is nn.Conv2d
    if pretrained:
        for name, param in net.named_parameters():
            if not 'classifier' in name:
                param.requires_grad = False
    return net