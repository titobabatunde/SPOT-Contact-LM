import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dropout_rate=0):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size, stride, padding, bias=False)
        self.inorm = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.conv(x)
        out = self.inorm(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1, dropout_rate=0.5):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, stride, padding, bias=False)
        self.inorm1 = nn.InstanceNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, 1, padding, bias=False)
        self.inorm2 = nn.InstanceNorm2d(channels)
        
        # Optional downsampling on the skip connection
        self.downsample = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=stride, bias=False),
            nn.InstanceNorm2d(channels)
        ) if stride != 1 else None

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.inorm1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.inorm2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu1(out) 

        return out

class iResNet(nn.Module):
    def __init__(self, channels, num_residual_blocks, num_classes=11, regression=False):
        super(iResNet, self).__init__()
        self.initial_block = BasicBlock(channels)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_residual_blocks)]
        )
        self.ln = nn.InstanceNorm2d(channels)
        self.act = nn.ReLU()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Channels to num_classes
        self.fc1 = nn.Linear(channels, num_classes) 
        
        self.regression = regression
        if regression:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.initial_block(x)
        x = self.residual_blocks(x)
        x = self.ln(x)
        x = self.act(x)
        
        x = self.avgpool(x)
        # flattenin the features into a vector
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        
        if self.regression:
            x = self.sigmoid(x)
        
        return x

"""
# Instantiate the model
channels = 64 
num_residual_blocks = 12
num_classes = 11  # The number of classes for classification
regression = False  # Set to True if the model is used for regression

# how the model should be called here
network = iResNet(channels, num_residual_blocks, num_classes=num_classes, regression=regression)
"""
