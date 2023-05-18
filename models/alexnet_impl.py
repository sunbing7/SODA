from torchvision.models import alexnet as alexnet_tv
import torch.nn as nn


def alexnet(num_classes=10, pretrained=1):
    net = alexnet_tv(pretrained=pretrained)
    if pretrained:
        for param in net.parameters():
            param.requires_grad = False

    net.classifier[6] = nn.Linear(4096, num_classes)    # nn.Linear in pytorch is a fully connected layer
                                                        # The convolutional layer is nn.Conv2d
    return net