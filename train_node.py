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
from sklearn.model_selection import train_test_split
from dataloader import Dataset_Loader

# from sklearn.model_selection import StratifiedSampler

# def get_class_counts(df):
#     grp = df.groupby

# def get_class_proportions(df):
#     class_counts = get_class_counts(df)
#     return {round(val[1]/df.shape[0],4) for val in class_counts.items()}

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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=50000, shuffle=True, num_workers=4)

# x = map(torch.tensor, zip(trainloader))
# print(trainloader[0].shape)

it = iter(trainloader)
# print(next(it)[0][0].shape)
x = next(it)[0]
y = torch.tensor(x)
print(torch.tensor(x).shape)

# train_dataset = Dataset_Loader(root='./data/',size=4000,isTrain='True')
# train_loader = DataLoader(train_dataset,batch_size=2,shuffle=True)

# valid_dataset = Dataset_Loader(root='./data/',size=1000,isTrain='False')
# valid_loader = DataLoader(valid_dataset,batch_size=2,shuffle=True)


# for i,data in enumerate(train_loader):
#     img = data['img'].reshape(-1,3,32,32)
#     labels = data['labels'].reshape(-1,1)


# class_labels = trainset.targets

# train_idx, valid_idx= train_test_split(
# np.arange(len(class_labels)),
# test_size=0.2,
# shuffle=True,
# stratify=class_labels)


# new_train_set = torch.utils.data.Subset(trainset, train_idx)
# new_val_set = torch.utils.data.Subset(trainset, valid_idx)

# print(new_train_set.__getitem__(0)[1])


# train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
# valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)


# train_batch_sampler = StratifiedSampler(class_labels, 10000)


# print(len(train_idx))
# print(train_idx)

# train_loader = torch.utils.data.DataLoader(trainset, batch_size=50000, sampler=train_sampler)
# valid_loader = torch.utils.data.DataLoader(trainset, batch_size=10000, sampler=valid_sampler)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# for batch_idx, (inputs, targets) in enumerate(train_loader):
#     inputs.to(device)
#     targets.to(device)
#     class_bit = [1,1,1,1,1,1,1,1,1,1]
#     node_num = 1
#     main_node = Node(inputs, targets, class_bit, 0, node_num, 10, device)
#     main_node.work()

# print(len(train_sampler))
# print(len(valid_loader.dataset))

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# train_sample_input = None
# train_sample_label = None
# val_sample_input = None
# val_sample_label = None
# class_count = [0]*10
# for batch_idx, (inputs, targets) in enumerate(trainloader):
#     for i in range(50000):
#         class_count[targets[i].item()] += 1
#         print(i)
#         if class_count[targets[i].item()] > 4000:
#             if val_sample_input is None:
#                 val_sample_input = inputs[[i],:,:,:]
#                 val_sample_label = targets[[i],]
#             else:
#                 val_sample_input = torch.cat((val_sample_input, inputs[[i],:,:,:]))
#                 val_sample_label = torch.cat((val_sample_label, targets[[i],]))
#         else:
#             if train_sample_input is None:
#                 train_sample_input = inputs[[i],:,:,:]
#                 train_sample_label = targets[[i],]
#             else:
#                 train_sample_input = torch.cat((train_sample_input, inputs[[i],:,:,:]))
#                 train_sample_label = torch.cat((train_sample_label, targets[[i],]))

# print(train_sample_input.shape)
# print(train_sample_label.shape)

# print(val_sample_input.shape)
# print(val_sample_label.shape)
# inputs.to(device)
# targets.to(device)
# class_bit = [1,1,1,1,1,1,1,1,1,1]
# node_num = 1
# main_node = Node(inputs, targets, class_bit, 0, node_num, 10, device)
# main_node.work()
