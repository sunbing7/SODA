# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Creating a CNN class
class ConvNeuralNet(nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, num_classes, bn=nn.BatchNorm2d):
        super(ConvNeuralNet, self).__init__()
        self.conv2d_1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1), nn.ReLU())
        self.conv2d_2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
                                      nn.ReLU())
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.2)

        self.conv2d_3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), nn.ReLU())
        self.conv2d_4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                      nn.ReLU())
        self.max_pool2= nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.3)

        self.conv2d_5 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), nn.ReLU())
        self.conv2d_6 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                      nn.ReLU())
        if bn:
            self.bn = bn(128)
        else:
            self.bn = None
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(p=0.4)

        self.fc1 = nn.Linear(2048, 512)
        self.relu1 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)
        self.act = nn.Softmax(dim=1)



    # Progresses data across layers
    def forward(self, x):
        out = self.conv2d_1(x)
        out = self.conv2d_2(out)

        activation1 = out
        out = self.max_pool1(out)
        out = self.dropout1(out)


        out = self.conv2d_3(out)
        out = self.conv2d_4(out)
        activation2 = out
        out = self.max_pool2(out)
        out = self.dropout2(out)

        out = self.conv2d_5(out)
        out = self.conv2d_6(out)
        if self.bn:
            out = self.bn(out)
        activation3 = out
        out = self.max_pool3(out)
        out = self.dropout3(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu1(out)
        out = self.dropout4(out)
        out = self.fc2(out)
        out = self.act(out)

        #return activation1, activation2, activation3, out
        return out


def cnn(**kwargs):
    """
    Constructs a cnn model.
    """
    return ConvNeuralNet(**kwargs)