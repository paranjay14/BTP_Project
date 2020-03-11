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
from sklearn.model_selection import train_test_split
from new_node import Node
import cv2
from sklearn.model_selection import train_test_split
# from queue import Queue

class Tree:
	def __init__(self, device, maxDepth=4, dominanceThreshold=0.95, classThreshold=2, dataNumThreshold=100, numClasses=10):
		self.maxDepth=maxDepth                         # depth threshold
		self.dominanceThreshold=dominanceThreshold     # threshold on class dominance
		self.classThreshold=classThreshold             # threshold on number of class in a node 
		self.nodeArray=None
		self.dataNumThreshold=dataNumThreshold         # threshold on total input images to a node
		self.numClasses = numClasses
		self.device = device
		self.root = None
		# self.maxNumberOfNodes = maxNumberOfNodes
	
	def tree_traversal(self, trainInputDict, valInputDict):
		nodeId = 1
		rootNode = Node(0, nodeId=nodeId, device=self.device, isTrain=True)
		rootNode.setInput(trainInputDict, valInputDict, self.numClasses)
		lTrainDict, lValDict, rTrainDict, rValDict = rootNode.work()

		# rootNode = Node(0, nodeId, self.device, True)
		# rootNode.setInput()

		# q = Queue(maxsize=self.maxNumberOfNodes)
		# queue = []
		# nodeNumbers = 1
		# newNode = Node(0, nodeNumbers, 10, self.numClasses, self.device, True)
		# queue.append(newNode)
		# while queue:
		#     q = queue.pop(0)
		#     q.work()
			
			
def load_image(path):
	img = cv2.imread(path)
	img = img.astype(np.float32)
	img = torch.from_numpy(img).permute(2,0,1)	

	return img


def loadDictionaries(rootPath):
	trainInputDict = {}
	valInputDict = {}
	trainData = torch.empty(1,3,32,32)
	valData = torch.empty(1,3,32,32)
	trainlabels = []
	vallabels = []
	# index = 0
	#change range index
	for index in range(10):
		for label in range(10):
			path = rootPath + str(label) + "/" + str(index) + ".jpg"
			img = load_image(path)
			img = torch.unsqueeze(img,0)
			if index == 0:
				trainData = img
			else:
				trainData = torch.cat((trainData, img), 0)
			trainlabels.append(label)
	
	trainLabels = torch.tensor(trainlabels)

	#change range index
	for index in range(10, 20):
		for label in range(10):
			path = rootPath + str(label) + "/" + str(index) + ".jpg"
			img = load_image(path)
			img = torch.unsqueeze(img, 0)
			if index == 0:
				valData = img
			else:
				valData = torch.cat((valData, img), 0)
			vallabels.append(label)
	
	valLabels = torch.tensor(vallabels)

	trainInputDict["data"] = trainData
	trainInputDict["label"] = trainLabels

	valInputDict["data"] = valData
	valInputDict["label"] = valLabels

	return trainInputDict, valInputDict


def loadNewDictionaries():
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
	class_labels = trainset.targets

	train_idx, valid_idx= train_test_split(
	np.arange(len(class_labels)),
	test_size=0.2,
	shuffle=True,
	stratify=class_labels)


	train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
	valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)


	# train_batch_sampler = StratifiedSampler(class_labels, 10000)


	# print(len(train_idx))
	# print(train_idx)

	train_loader = torch.utils.data.DataLoader(trainset, batch_size=50000, sampler=train_sampler)
	valid_loader = torch.utils.data.DataLoader(trainset, batch_size=10000, sampler=valid_sampler)


	# trainloader = torch.utils.data.DataLoader(trainset, batch_size=40000, shuffle=True, num_workers=4)

	iterator = iter(train_loader)
	c1 = next(iterator)
	trainData = c1[0].clone().detach()
	trainLabels = c1[1].clone().detach()

	vIterator = iter(valid_loader)
	c2 = next(vIterator)
	valData = c2[0].clone().detach()
	valLabels = c2[1].clone().detach()
	return {"data":trainData, "label":trainLabels}, {"data":valData, "label":valLabels}


			
if __name__ == '__main__':
	# trainImages, trainLabels, valImages, valLabels = loadDataset()
	# print(len(trainImages))
	# it = iter(valLoader)
	# print(len(next(it)[0][0]))

	rootPath = "./data/"
	
	trainInputDict, valInputDict = loadNewDictionaries()
	# print(trainInputDict["data"][0].shape)

		
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# tree = Tree(device, maxDepth=1, classThreshold = 2, dataNumThreshold = 1, numClasses = 10)
	# tree.tree_traversal(trainInputDict, valInputDict)


