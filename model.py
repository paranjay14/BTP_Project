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
# from prev_new_node import Node
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from getOptions import getOptions
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
	

	def checkLeafNodes(self, handleLeafDict):
		isLeafLeft=False
		isLeafRight=False
		isemptyNodeLeft=False
		isemptyNodeRight=False
		leftLeafClass = -1
		rightLeafClass = -1

		############# maxDepth #############
		if handleLeafDict["lvl"]==self.maxDepth:
			isLeafLeft=True
			isLeafRight=True
			print("MAX DEPTH REACHED", self.maxDepth)

		############# dataNumThreshold #############
		if handleLeafDict["leftDataNum"] <= self.dataNumThreshold:
			isLeafLeft=True
			print("DATANUM THRESHOLD REACHED IN LEFT", handleLeafDict["leftDataNum"], self.dataNumThreshold)
		if handleLeafDict["rightDataNum"] <= self.dataNumThreshold:
			isLeafRight=True		
			print("DATANUM THRESHOLD REACHED IN RIGHT", handleLeafDict["rightDataNum"], self.dataNumThreshold)

		############# classThreshold #############
		#TODO : HANDLE 0, 1 & 2 cases
		if handleLeafDict["noOfLeftClasses"]==0 or handleLeafDict["noOfRightClasses"]==0:
			if handleLeafDict["noOfLeftClasses"]==0:
				isemptyNodeLeft=True
				print("CLASS THRESHOLD REACHED 0 IN LEFT", handleLeafDict["noOfLeftClasses"], self.classThreshold)
			if handleLeafDict["noOfRightClasses"]==0:
				isemptyNodeRight=True
				print("CLASS THRESHOLD REACHED 0 IN RIGHT", handleLeafDict["noOfRightClasses"], self.classThreshold)

		elif handleLeafDict["noOfLeftClasses"]==1 or handleLeafDict["noOfRightClasses"]==1:
			if handleLeafDict["noOfLeftClasses"]==1:
				isLeafLeft=True
				leftLeafClass=handleLeafDict["maxLeftClassIndex"]
				print("CLASS THRESHOLD REACHED 1 IN LEFT", handleLeafDict["noOfLeftClasses"], self.classThreshold, leftLeafClass)

			if handleLeafDict["noOfRightClasses"]==1:
				isLeafRight=True
				rightLeafClass=handleLeafDict["maxRightClassIndex"]
				print("CLASS THRESHOLD REACHED 1 IN RIGHT", handleLeafDict["noOfRightClasses"], self.classThreshold, rightLeafClass)

		elif handleLeafDict["noOfLeftClasses"]==2 or handleLeafDict["noOfRightClasses"]==2:
			if handleLeafDict["noOfLeftClasses"]==2:
				isLeafLeft=True
				print("CLASS THRESHOLD REACHED 2 IN LEFT", handleLeafDict["noOfLeftClasses"], self.classThreshold)

			if handleLeafDict["noOfRightClasses"]==2:
				isLeafRight=True
				print("CLASS THRESHOLD REACHED 2 IN RIGHT", handleLeafDict["noOfRightClasses"], self.classThreshold)


		############# dominanceThreshold #############
		if handleLeafDict["maxLeft"] >= self.dominanceThreshold:
			isLeafLeft=True
			leftLeafClass=handleLeafDict["maxLeftClassIndex"]
			print("DOMINANCE THRESHOLD REACHED IN LEFT", handleLeafDict["maxLeft"], self.dominanceThreshold, leftLeafClass)

		if handleLeafDict["maxRight"] >= self.dominanceThreshold:
			isLeafRight=True
			rightLeafClass=handleLeafDict["maxRightClassIndex"]
			print("DOMINANCE THRESHOLD REACHED IN RIGHT", handleLeafDict["maxRight"], self.dominanceThreshold, rightLeafClass)


		return isLeafLeft, isLeafRight, isemptyNodeLeft, isemptyNodeRight, leftLeafClass, rightLeafClass


	def tree_traversal(self, trainInputDict, valInputDict):
		rootNode = Node(parentId=0, nodeId=1, device=self.device, isTrain=True, level=0)
		if self.maxDepth == 0:
			rootNode.setInput(trainInputDict=trainInputDict, valInputDict=valInputDict, numClasses=self.numClasses, giniValue=0.9, isLeaf=True, leafClass=-1, lchildId=-1, rchildId=-1)
		else:
			rootNode.setInput(trainInputDict=trainInputDict, valInputDict=valInputDict, numClasses=self.numClasses, giniValue=0.9, isLeaf=False, leafClass=-1, lchildId=-1, rchildId=-1)

		self.nodeArray = []
		self.nodeArray.append(rootNode)
		start = 0
		end = 1
		while start != end:
			node = self.nodeArray[start]
			print("Running nodeId: ", node.nodeId)
			start+=1
			if not node.isLeaf:
				# lTrainDict, lValDict, rTrainDict, rValDict, giniLeftRatio, giniRightRatio, noOfLeftClasses, noOfRightClasses = node.work()
				lTrainDict, lValDict, rTrainDict, rValDict, giniLeftRatio, giniRightRatio, handleLeafDict = node.workTrain()
			else:
				# node.work()
				node.workTrain()

			if not node.isLeaf:
				isLeafLeft, isLeafRight, isemptyNodeLeft, isemptyNodeRight, leftLeafClass, rightLeafClass = self.checkLeafNodes(handleLeafDict)
				ParentNodeDict = torch.load(options.ckptDir+'/node_'+str(node.nodeId)+'.pth')['nodeDict']

				if not isemptyNodeLeft:
					end += 1
					lNode = Node(node.nodeId, end, self.device, True, node.level+1)
					if not isLeafLeft:
						lNode.setInput(lTrainDict, lValDict, handleLeafDict["noOfLeftClasses"], giniLeftRatio, False, leftLeafClass, -1, -1)
					else:
						lNode.setInput(lTrainDict, lValDict, handleLeafDict["noOfLeftClasses"], giniLeftRatio, True, leftLeafClass, -1, -1)
					self.nodeArray.append(lNode)
					ParentNodeDict['lchildId'] = lNode.nodeId

				if not isemptyNodeRight:
					end += 1
					rNode = Node(node.nodeId, end, self.device, True, node.level+1)
					if not isLeafRight:
						rNode.setInput(rTrainDict, rValDict, handleLeafDict["noOfRightClasses"], giniRightRatio, False, rightLeafClass, -1, -1)
					else:
						rNode.setInput(rTrainDict, rValDict, handleLeafDict["noOfRightClasses"], giniRightRatio, True, rightLeafClass, -1, -1)
					self.nodeArray.append(rNode)
					ParentNodeDict['rchildId'] = rNode.nodeId

				torch.save({
					'nodeDict':ParentNodeDict,
					}, options.ckptDir+'/node_'+str(node.nodeId)+'.pth')

				# end += 2
		

	def testTraversal(self, testInputDict):
		nodeId=1
		ckptRoot = torch.load(options.ckptDir+'/node_cnn_'+str(nodeId)+'.pth')['labelMap']
		noOfClasses = len(ckptRoot)
		rootNodeDict = torch.load(options.ckptDir+'/node_'+str(nodeId)+'.pth')['nodeDict']
		isLeafRoot = rootNodeDict['isLeaf']
		leftChildId = rootNodeDict['lchildId']
		rightChildId = rootNodeDict['rchildId']
		if rootNodeDict['level']>=self.maxDepth:
			isLeafRoot=True
			leftChildId=-1
			rightChildId=-1
		rootNode = Node(parentId=rootNodeDict['parentId'], nodeId=rootNodeDict['nodeId'], device=self.device, isTrain=False, level=rootNodeDict['level'])
		rootNode.setInput(trainInputDict=testInputDict, valInputDict={}, numClasses=noOfClasses, giniValue=0.9, isLeaf=isLeafRoot, leafClass=rootNodeDict['leafClass'], lchildId=leftChildId, rchildId=rightChildId)
		
		testPredDict = {}
		testPredDict['actual'] = torch.rand(0)
		testPredDict['pred'] = torch.rand(0)
		testPredDict['actual'] = testPredDict['actual'].long()
		testPredDict['pred'] = testPredDict['pred'].long()
		testPredDict['actual'] = testPredDict['actual'].to(self.device)
		testPredDict['pred'] = testPredDict['pred'].to(self.device)

		torch.save({
					'testPredDict':testPredDict,
					}, options.ckptDir+'/testPred.pth')

		q = []
		q.append(rootNode)
		start = 0
		end = 1
		while start != end:
			node = q[start]
			start+=1
			if not node.isLeaf:
				# lTrainDict, rTrainDict,  giniLeftRatio, giniRightRatio, noOfLeftClasses, noOfRightClasses = node.work()
				lTrainDict, rTrainDict,  giniLeftRatio, giniRightRatio, noOfLeftClasses, noOfRightClasses = node.workTest()
			else:
				# node.work()
				node.workTest()
			if not node.isLeaf:

				if not (node.lchildId == -1):
					leftNodeDict = torch.load(options.ckptDir+'/node_'+str(node.lchildId)+'.pth')['nodeDict']
					noOfLeftClasses = 1		
					if (leftNodeDict['leafClass'] == -1):
						ckptLeft = torch.load(options.ckptDir+'/node_cnn_'+str(node.lchildId)+'.pth')['labelMap']
						noOfLeftClasses = len(ckptLeft)
 
					lNode = Node(node.nodeId, node.lchildId, self.device, False, leftNodeDict['level'])
					isLeafLeft = leftNodeDict['isLeaf']
					leftChildId = leftNodeDict['lchildId']
					rightChildId = leftNodeDict['rchildId']
					if leftNodeDict['level']>=self.maxDepth:
						isLeafLeft=True
						leftChildId=-1
						rightChildId=-1
					lNode.setInput(lTrainDict, {}, noOfLeftClasses, giniLeftRatio, isLeafLeft, leftNodeDict['leafClass'], leftChildId, rightChildId)
					q.append(lNode)
					end+=1
				
				if not (node.rchildId == -1):
					rightNodeDict = torch.load(options.ckptDir+'/node_'+str(node.rchildId)+'.pth')['nodeDict']
					noOfRightClasses=1
					if (rightNodeDict['leafClass'] == -1):		
						ckptRight = torch.load(options.ckptDir+'/node_cnn_'+str(node.rchildId)+'.pth')['labelMap']
						noOfRightClasses = len(ckptRight)
						
					rNode = Node(node.nodeId, node.rchildId, self.device, False, rightNodeDict['level'])
					isLeafRight = rightNodeDict['isLeaf']
					leftChildId = rightNodeDict['lchildId']
					rightChildId = rightNodeDict['rchildId']
					if rightNodeDict['level']>=self.maxDepth:
						isLeafRight=True
						leftChildId=-1
						rightChildId=-1
					rNode.setInput(rTrainDict, {}, noOfRightClasses, giniRightRatio, isLeafRight, rightNodeDict['leafClass'], leftChildId, rightChildId)
					q.append(rNode)
					end+=1

				print ('Nodes sizes = ', noOfLeftClasses, noOfRightClasses)


		ckpt = torch.load(options.ckptDir+'/testPred.pth')
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

	if not os.path.isdir(options.ckptDir+'/'):
		# print("naf")
		os.mkdir(options.ckptDir+'/')


	print('==> Preparing data..')
	transform_train = transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	transform_test = transforms.Compose([
		transforms.ToTensor(),
		# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
	class_labels = trainset.targets

	# train_idx, valid_idx= train_test_split(
	# np.arange(len(class_labels)),
	# test_size=0.2,
	# shuffle=True,
	# stratify=class_labels)
	train_idx, valid_idx= train_test_split(
	np.arange(len(class_labels)),
	test_size=0.0002,
	shuffle=True,
	stratify=class_labels)

	train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
	valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

	# train_batch_sampler = StratifiedSampler(class_labels, 10000)

	# print(len(train_idx))
	# print(train_idx)

	# train_loader = torch.utils.data.DataLoader(trainset, batch_size=40000, sampler=train_sampler, num_workers=4)
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=49900, sampler=train_sampler, num_workers=4)
	# valid_loader = torch.utils.data.DataLoader(trainset, batch_size=10000, sampler=valid_sampler, num_workers=4)
	valid_loader = torch.utils.data.DataLoader(trainset, batch_size=100, sampler=valid_sampler, num_workers=4)

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
	options = getOptions(sys.argv[1:])

	trainInputDict, valInputDict, testInputDict = loadNewDictionaries()
	print("len(trainInputDict[\"data\"]): ",len(trainInputDict["data"]), ",  len(valInputDict[\"data\"]): ",len(valInputDict["data"]), ",  len(testInputDict[\"data\"]): ",len(testInputDict["data"]))		
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	tree = Tree(device, maxDepth=options.maxDepth, classThreshold = 2, dataNumThreshold = 1, numClasses = 10)
	
	if options.trainFlg == True:
		tree.tree_traversal(trainInputDict, valInputDict)
		# tree.tree_traversal(valInputDict, valInputDict)
	tree.testTraversal(testInputDict)

