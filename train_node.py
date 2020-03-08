import argparse
import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os.path
import torch.nn as nn
import torchvision.transforms as transforms
# from model import *
from node import *
import os

if not os.path.isdir('data/'):
    # print("naf")
    os.mkdir('data/')

if not os.path.isdir('ckpt/'):
    # print("naf")
    os.mkdir('ckpt/')

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10000, shuffle=True, num_workers=4)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for batch_idx, (inputs, targets) in enumerate(trainloader):
    print(inputs.shape)
# (inputs, targets) = trainloader.
    # one_hot = torch.zeros([1000, 10])
    # for i in range(1000):
        # j = targets[i].item()
        # one_hot[i][j-1] = 1
    inputs.to(device)
    targets.to(device)
    # one_hot.to(device)
    # print(inputs.shape)
    # print(one_hot.shape)
    class_bit = [1,1,1,1,1,1,1,1,1,1]
    node_num = 1
    # print(inputs)
    # print(targets)
    main_node = Node(inputs, targets, class_bit, 0, node_num, 10, device)
    main_node.work()
