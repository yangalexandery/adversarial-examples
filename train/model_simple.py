import torch
import torch.nn as nn
import torch.nn.functional as F
# from utee import misc
from collections import OrderedDict


__all__ = ['SimpleConvNet']

# https://arxiv.org/pdf/1412.6806.pdf

class SimpleConvNet(nn.Module):

    def __init__(self, num_classes=10, transform_input=False):
        super(SimpleConvNet, self).__init__()
        self.transform_input = transform_input
        self.conv1 = nn.Conv2d(3, 96, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(96, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(192, 192, kernel_size=1)
        self.conv5 = nn.Conv2d(192, 10, kernel_size=1)
        # self.fc1 = nn.Linear(10*7*7, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.avg_pool2d(x, kernel_size=7)
        # x = x.view(-1, 192 * 7 * 7)
        x = x.view(x.size(0), -1)#10 * 7 * 7)
        # x = self.fc1(x)
        return x