import argparse
import sys
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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# from resources.plotcm import plot_confusion_matrix
# from queue import Queue

class Tree:
	def __init__(self, device, maxDepth=1, dominanceThreshold=0.95, classThreshold=2, dataNumThreshold=100, numClasses=10):
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
		rootNode = Node(0, nodeId=nodeId, device=self.device, isTrain=True, level=0)
		# rootNode = Node(0, nodeId=nodeId, device=self.device, isTrain=False)
		rootNode.setInput(trainInputDict, valInputDict, self.numClasses, 0.9, False)
		# lTrainDict, lValDict, rTrainDict, rValDict, giniLeftRatio, giniRightRatio = rootNode.work()

		# q = Queue(maxsize=self.maxNumberOfNodes)
		# queue = []
		self.nodeArray = []
		nodeNumbers = 1
		# newNode = Node(0, nodeNumbers, 10, self.numClasses, self.device, True)
		self.nodeArray.append(rootNode)
		start = 0
		end = 1
		while start != end:
			node = self.nodeArray[start]
			print("Running nodeId: ", node.nodeId)
			start+=1
			if not node.isLeaf:
				lTrainDict, lValDict, rTrainDict, rValDict, giniLeftRatio, giniRightRatio, noOfLeftClasses, noOfRightClasses = node.work()
			else:
				node.work()

			if not node.isLeaf:
				lNode = Node(node.nodeId, end+1, self.device, True, node.level+1)
				rNode = Node(node.nodeId, end+2, self.device, True, node.level+1)

				if node.level + 1 >= self.maxDepth:
					lNode.setInput(lTrainDict, lValDict, noOfLeftClasses, giniLeftRatio, True)
					rNode.setInput(rTrainDict, rValDict, noOfRightClasses, giniRightRatio, True)
				else:
					lNode.setInput(lTrainDict, lValDict, noOfLeftClasses, giniLeftRatio, False)
					rNode.setInput(rTrainDict, rValDict, noOfRightClasses, giniRightRatio, False)

				self.nodeArray.append(lNode)
				self.nodeArray.append(rNode)
				end += 2
			# end = min(end, 3)

		

	def testTraversal(self, valInputDict):
		rootNode = Node(0, nodeId=1, device=self.device, isTrain=False, level=0)
		if self.maxDepth == 0:
			rootNode.setInput(valInputDict, {}, self.numClasses, 0.9, True)
		else:
			rootNode.setInput(valInputDict, {}, self.numClasses, 0.9, False)
		# lTrainDict, rTrainDict, giniLeftRatio, giniRightRatio, noOfLeftClasses, noOfRightClasses = rootNode.work()
		
		testPredDict = {}
		testPredDict['actual'] = torch.rand(0)
		testPredDict['pred'] = torch.rand(0)
		testPredDict['actual'] = testPredDict['actual'].long()
		testPredDict['pred'] = testPredDict['pred'].long()
		testPredDict['actual'] = testPredDict['actual'].to(self.device)
		testPredDict['pred'] = testPredDict['pred'].to(self.device)

		torch.save({
					'testPredDict':testPredDict,
					}, 'ckpt/testPred.pth')

		q = []
		q.append(rootNode)
		start = 0
		end = 1
		while start != end:
			node = q[start]
			start+=1
			if not node.isLeaf:
				lTrainDict, rTrainDict,  giniLeftRatio, giniRightRatio, noOfLeftClasses, noOfRightClasses = node.work()
			else:
				node.work()
			if not node.isLeaf:
				lNode = Node(node.nodeId, end+1, self.device, False, node.level+1)
				rNode = Node(node.nodeId, end+2, self.device, False, node.level+1)

				if node.level + 1 >= self.maxDepth:
					lNode.setInput(lTrainDict, {}, noOfLeftClasses, giniLeftRatio, True)
					rNode.setInput(rTrainDict, {}, noOfRightClasses, giniRightRatio, True)
				else:
					lNode.setInput(lTrainDict, {}, noOfLeftClasses, giniLeftRatio, False)
					rNode.setInput(rTrainDict, {}, noOfRightClasses, giniRightRatio, False)

				q.append(lNode)
				q.append(rNode)
				end += 2

		ckpt = torch.load('ckpt/testPred.pth')
		testPredDict = ckpt['testPredDict']
		testPredDict['actual'] = testPredDict['actual'].to("cpu")
		testPredDict['pred'] = testPredDict['pred'].to("cpu")
		cm = confusion_matrix(testPredDict['actual'], testPredDict['pred'])
		print(cm)
		print()
		correct = testPredDict['pred'].eq(testPredDict['actual']).sum().item()
		total = len(testPredDict['actual'])
		print('Acc: %.3f'% (100.*correct/total))

			
			
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

	train_loader = torch.utils.data.DataLoader(trainset, batch_size=40000, sampler=train_sampler, num_workers=4)
	valid_loader = torch.utils.data.DataLoader(trainset, batch_size=10000, sampler=valid_sampler, num_workers=4)


	# trainloader = torch.utils.data.DataLoader(trainset, batch_size=40000, shuffle=True, num_workers=4)

	iterator = iter(train_loader)
	c1 = next(iterator)
	trainData = c1[0].clone().detach()
	trainLabels = c1[1].clone().detach()

	vIterator = iter(valid_loader)
	c2 = next(vIterator)
	valData = c2[0].clone().detach()
	valLabels = c2[1].clone().detach()

	# print(trainData.shape)
	# print(valData.shape)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False)

	testIterator = iter(testloader)
	c3 = next(testIterator)
	testData = c3[0].clone().detach()
	testLabels = c3[1].clone().detach()

	return {"data":trainData, "label":trainLabels}, {"data":valData, "label":valLabels}, {"data":testData, "label":testLabels}



			
if __name__ == '__main__':
	# trainImages, trainLabels, valImages, valLabels = loadDataset()
	# print(len(trainImages))
	# it = iter(valLoader)
	# print(len(next(it)[0][0]))

	rootPath = "./data/"
	maxDepth = int(sys.argv[1])
	trainInputDict, valInputDict, testInputDict = loadNewDictionaries()
	# print(trainInputDict["data"][0].shape)

		
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# if torch.cuda.is_available():
	# 	print("cuda is_available")
	# else:
	# 	print("cpu pe chl rha :(")
	# print(device)
	# device = torch.device("cpu")
	tree = Tree(device, maxDepth=maxDepth, classThreshold = 2, dataNumThreshold = 1, numClasses = 10)
	# tree.tree_traversal(trainInputDict, valInputDict)
	# tree.tree_traversal(valInputDict, valInputDict)
	tree.testTraversal(testInputDict)


