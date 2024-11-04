import torch
from torch.amp import autocast

@torch.no_grad()
def calc_acc(model, loader, device):
    model.eval()
    test_corrects = 0
    test_total = 0
    for images, labels in iter(loader):
        images = images.to(device)
        labels = labels.to(device)
        with autocast(device_type='cuda', dtype=torch.float16):
            bag_logit = model.eval_forward(images)
            _, preds = torch.max(bag_logit, dim=1)
            correct = (preds == labels).sum()
            test_corrects += correct.item()
            test_total += preds.size(0)
    
    test_epoch_acc = (test_corrects / test_total) * 100
    print(f"Test Accuracy: {test_epoch_acc:.2f}%\n")

import random
from six.moves import urllib
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

class MILMNISTDataset(Dataset):
    def __init__(self, root, target_digits=[6, 8, 9], bag_size=64, train=True, transform=None, download=True):
        self.target_digits = target_digits
        self.bag_size = bag_size
        self.transform = transform

        # Load MNIST dataset
        self.mnist = datasets.MNIST(root=root, train=train, transform=transform, download=download)

        # Separate indices for target and non-target digits
        self.target_indices = []
        for target_digit in target_digits:
            self.target_indices += [ [i for i, label in enumerate(self.mnist.targets) if label == target_digit] ]
        self.non_target_indices = [i for i, label in enumerate(self.mnist.targets) if label not in target_digits]

    def __len__(self):
        # Define the number of bags; can be arbitrary
        length = 0
        for tranget_index in self.target_indices:
            length += len(tranget_index)
        return length

    def __getitem__(self, idx):
        instances = []

        # Ensure at least one target digit in the bag
        target_digit = random.choice(list(range(len(self.target_indices))))
        target_idx = random.choice(self.target_indices[target_digit])
        image, label = self.mnist[target_idx]
        instances += [image]
        bag_label = target_digit

        # Fill the rest with target and non-target digits
        non_target_idxs = random.choices(self.non_target_indices + self.target_indices[target_digit], k=self.bag_size-1)
        for non_target_idx in non_target_idxs:
            img, lbl = self.mnist[non_target_idx]
            instances += [img]


        # Shuffle instances within the bag
        random.shuffle(instances)

        # Stack instances into a tensor
        instances = torch.stack(instances)  # Shape: [bag_size, H, W]
        instances = instances.repeat(1, 3, 1, 1)

        return instances, bag_label


import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class Bottleneck_Baseline(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_Baseline, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet_Baseline(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet_Baseline, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

    def forward(self, x):
        return self.forward_features(x)

def resnet50_baseline(pretrained=False):
    """Constructs a Modified ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_Baseline(Bottleneck_Baseline, [3, 4, 6, 3])
    if pretrained:
        model = load_pretrained_weights(model, 'resnet50')
    return model

def load_pretrained_weights(model, name):
    pretrained_dict = model_zoo.load_url(model_urls[name])
    model.load_state_dict(pretrained_dict, strict=False)
    return model
