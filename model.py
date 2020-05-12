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
from new_node import myNode
# from prev_new_node import Node
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from getOptions import getOptions
from pptree import *
import random
import time
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
		# HANDLE 0, 1 & 2 cases
		if handleLeafDict["noOfLeftClasses"]==0:
			isemptyNodeLeft=True
			print("CLASS THRESHOLD REACHED 0 IN LEFT", handleLeafDict["noOfLeftClasses"], self.classThreshold)

		elif handleLeafDict["noOfLeftClasses"]==1:
			isLeafLeft=True
			leftLeafClass=handleLeafDict["maxLeftClassIndex"]
			print("CLASS THRESHOLD REACHED 1 IN LEFT", handleLeafDict["noOfLeftClasses"], self.classThreshold, leftLeafClass)

		# elif handleLeafDict["noOfLeftClasses"]==2:
		# 	isLeafLeft=True
		# 	print("CLASS THRESHOLD REACHED 2 IN LEFT", handleLeafDict["noOfLeftClasses"], self.classThreshold)


		if handleLeafDict["noOfRightClasses"]==0:
			isemptyNodeRight=True
			print("CLASS THRESHOLD REACHED 0 IN RIGHT", handleLeafDict["noOfRightClasses"], self.classThreshold)

		elif handleLeafDict["noOfRightClasses"]==1:
			isLeafRight=True
			rightLeafClass=handleLeafDict["maxRightClassIndex"]
			print("CLASS THRESHOLD REACHED 1 IN RIGHT", handleLeafDict["noOfRightClasses"], self.classThreshold, rightLeafClass)

		# elif handleLeafDict["noOfRightClasses"]==2:
		# 	isLeafRight=True
		# 	print("CLASS THRESHOLD REACHED 2 IN RIGHT", handleLeafDict["noOfRightClasses"], self.classThreshold)


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


	def tree_traversal(self, trainInputDict, valInputDict, resumeTrain, resumeFromNodeId):
		rootNode = myNode(parentId=0, nodeId=1, device=self.device, isTrain=True, level=0, parentNode=None)
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
				if (resumeTrain) and (node.nodeId<resumeFromNodeId):
					lTrainDict, lValDict, rTrainDict, rValDict, giniLeftRatio, giniRightRatio, handleLeafDict = node.workTest()
				else:
					lTrainDict, lValDict, rTrainDict, rValDict, giniLeftRatio, giniRightRatio, handleLeafDict = node.workTrain()
			else:
				if (resumeTrain) and (node.nodeId<resumeFromNodeId):
					node.workTest()
				else:
					node.workTrain()

			if not node.isLeaf:
				isLeafLeft, isLeafRight, isemptyNodeLeft, isemptyNodeRight, leftLeafClass, rightLeafClass = self.checkLeafNodes(handleLeafDict)
				ParentNodeDict = torch.load(options.ckptDir+'/node_'+str(node.nodeId)+'.pth')['nodeDict']

				if not isemptyNodeLeft:
					end += 1
					lNode = myNode(node.nodeId, end, self.device, True, node.level+1, node)
					if not isLeafLeft:
						lNode.setInput(lTrainDict, lValDict, handleLeafDict["noOfLeftClasses"], giniLeftRatio, False, leftLeafClass, -1, -1)
					else:
						lNode.setInput(lTrainDict, lValDict, handleLeafDict["noOfLeftClasses"], giniLeftRatio, True, leftLeafClass, -1, -1)
					self.nodeArray.append(lNode)
					ParentNodeDict['lchildId'] = lNode.nodeId

				if not isemptyNodeRight:
					end += 1
					rNode = myNode(node.nodeId, end, self.device, True, node.level+1,node)
					if not isLeafRight:
						rNode.setInput(rTrainDict, rValDict, handleLeafDict["noOfRightClasses"], giniRightRatio, False, rightLeafClass, -1, -1)
					else:
						rNode.setInput(rTrainDict, rValDict, handleLeafDict["noOfRightClasses"], giniRightRatio, True, rightLeafClass, -1, -1)
					self.nodeArray.append(rNode)
					ParentNodeDict['rchildId'] = rNode.nodeId

				ParentNodeDict['giniGain'] = handleLeafDict["giniGain"]
				torch.save({
					'nodeDict':ParentNodeDict,
					}, options.ckptDir+'/node_'+str(node.nodeId)+'.pth')


	def testTraversal(self, testInputDict):
		print("TESTING STARTS")
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
		rootNode = myNode(parentId=rootNodeDict['parentId'], nodeId=rootNodeDict['nodeId'], device=self.device, isTrain=False, level=rootNodeDict['level'], parentNode=None)
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

		LevelDict = {}
		LevelDict['levelAcc'] = {}
		LevelDict['leafAcc'] = [0,0]
		torch.save({
			'levelDict':LevelDict,
			}, options.ckptDir+'/level.pth')

		prevLvl=-1
		q = []
		q.append(rootNode)
		start = 0
		end = 1
		while start != end:
			node = q[start]
			curLvl=node.level
			if curLvl>prevLvl:
				LevelDict = torch.load(options.ckptDir+'/level.pth')['levelDict']
				LevelDict['levelAcc'][curLvl] = LevelDict['leafAcc'][:]
				torch.save({
					'levelDict':LevelDict,
					}, options.ckptDir+'/level.pth')
				prevLvl=curLvl

			start+=1
			if not node.isLeaf:
				lTrainDict, rTrainDict,  giniLeftRatio, giniRightRatio, noOfLeftClasses, noOfRightClasses = node.workTest()
			else:
				node.workTest()
			if not node.isLeaf:

				if not (node.lchildId == -1):
					leftNodeDict = torch.load(options.ckptDir+'/node_'+str(node.lchildId)+'.pth')['nodeDict']
					noOfLeftClasses = 1		
					if (leftNodeDict['leafClass'] == -1):
						ckptLeft = torch.load(options.ckptDir+'/node_cnn_'+str(node.lchildId)+'.pth')['labelMap']
						noOfLeftClasses = len(ckptLeft)
 
					lNode = myNode(node.nodeId, node.lchildId, self.device, False, leftNodeDict['level'],node)
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
						
					rNode = myNode(node.nodeId, node.rchildId, self.device, False, rightNodeDict['level'], node)
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
		print('Final Acc: %.3f'% (100.*correct/total))
		print()

		LevelDict = torch.load(options.ckptDir+'/level.pth')['levelDict']
		for i,val in enumerate(LevelDict['levelAcc'].items()):
			print('Level %d Acc: %.3f'% (val[0], 100.*val[1][0]/val[1][1]))
	
	def printTree(self):
		nodeId=1
		rootNodeDict = torch.load(options.ckptDir+'/node_'+str(nodeId)+'.pth')['nodeDict']
		rootNode = myNode(parentId=rootNodeDict['parentId'], nodeId=rootNodeDict['nodeId'], device=self.device, isTrain=False, level=rootNodeDict['level'], parentNode=None)

		q = []
		q.append(rootNode)
		start = 0
		end = 1
		while start != end:
			node = q[start]
			start+=1
			currNodeDict = torch.load(options.ckptDir+'/node_'+str(node.nodeId)+'.pth')['nodeDict']
			node.isLeaf = currNodeDict['isLeaf']
			node.lchildId = currNodeDict['lchildId']
			node.rchildId = currNodeDict['rchildId']
			node.level	= 	currNodeDict['level']
			node.leafClass = currNodeDict['leafClass']
			node.numClasses = currNodeDict['numClasses']
			node.numData = currNodeDict['numData']
			node.classLabels = currNodeDict['classLabels']
			node.giniGain = currNodeDict['giniGain']
			node.splitAcc = currNodeDict['splitAcc']
			node.nodeAcc = currNodeDict['nodeAcc']

			if not node.isLeaf:
				if not (node.lchildId == -1):
					lNode = myNode(node.nodeId, node.lchildId, self.device, False, node.level+1, node)
					q.append(lNode)
					end+=1

				if not (node.rchildId == -1):
					rNode = myNode(node.nodeId, node.rchildId, self.device, False, node.level+1, node)
					q.append(rNode)
					end+=1

				
		
		print()
		print_tree(rootNode, "children", "nodeId", horizontal=False)	
		print()	
		print_tree(rootNode, "children", "leafClass", horizontal=False)		
		print()	
		# print_tree(rootNode, "children", "isLeaf", horizontal=False)
		# print()	
		# print_tree(rootNode, "children", "numClasses", horizontal=False)
		# print()
		print_tree(rootNode, "children", "classLabels", horizontal=True)
		print()	
		print_tree(rootNode, "children", "splitAcc", horizontal=False)
		print()
		print_tree(rootNode, "children", "giniGain", horizontal=False)
		print()	
		print_tree(rootNode, "children", "numData", horizontal=False)
		print()	
		print_tree(rootNode, "children", "nodeAcc", horizontal=True)
		print()	

			
# def load_image(path):
# 	img = cv2.imread(path)
# 	img = img.astype(np.float32)
# 	img = torch.from_numpy(img).permute(2,0,1)	

# 	return img


# def loadDictionaries(rootPath):
# 	trainInputDict = {}
# 	valInputDict = {}
# 	trainData = torch.empty(1,3,32,32)
# 	valData = torch.empty(1,3,32,32)
# 	trainlabels = []
# 	vallabels = []
# 	# index = 0
# 	#change range index
# 	for index in range(10):
# 		for label in range(10):
# 			path = rootPath + str(label) + "/" + str(index) + ".jpg"
# 			img = load_image(path)
# 			img = torch.unsqueeze(img,0)
# 			if index == 0:
# 				trainData = img
# 			else:
# 				trainData = torch.cat((trainData, img), 0)
# 			trainlabels.append(label)
	
# 	trainLabels = torch.tensor(trainlabels)

# 	#change range index
# 	for index in range(10, 20):
# 		for label in range(10):
# 			path = rootPath + str(label) + "/" + str(index) + ".jpg"
# 			img = load_image(path)
# 			img = torch.unsqueeze(img, 0)
# 			if index == 0:
# 				valData = img
# 			else:
# 				valData = torch.cat((valData, img), 0)
# 			vallabels.append(label)
	
# 	valLabels = torch.tensor(vallabels)

# 	trainInputDict["data"] = trainData
# 	trainInputDict["label"] = trainLabels

# 	valInputDict["data"] = valData
# 	valInputDict["label"] = valLabels

# 	return trainInputDict, valInputDict


def loadNewDictionaries():

	torch.manual_seed(0)
	torch.cuda.manual_seed(0)
	np.random.seed(0)
	random.seed(0)

	torch.backends.cudnn.deterministic=True

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

	# train_batch_sampler = StratifiedSampler(class_labels, 10000)



	# '''   -->  PREPEND # FOR NO VALIDATION
	train_idx, valid_idx= train_test_split(
	np.arange(len(class_labels)),
	test_size=0.02,
	shuffle=True,
	stratify=class_labels)

	train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
	valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

	train_loader = torch.utils.data.DataLoader(trainset, batch_size=49000, sampler=train_sampler, num_workers=0)
	valid_loader = torch.utils.data.DataLoader(trainset, batch_size=1000, sampler=valid_sampler, num_workers=0)
	
	iterator = iter(train_loader)
	c1 = next(iterator)
	trainData = c1[0].clone().detach()
	trainLabels = c1[1].clone().detach()

	vIterator = iter(valid_loader)
	c2 = next(vIterator)
	valData = c2[0].clone().detach()
	valLabels = c2[1].clone().detach()
	

	'''
	train_loader = torch.utils.data.DataLoader(trainset, batch_size=50000, num_workers=0)
	iterator = iter(train_loader)
	c1 = next(iterator)
	trainData = c1[0].clone().detach()
	trainLabels = c1[1].clone().detach()

	valData=torch.empty(0)
	valLabels=torch.empty(0)
	# '''



	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False)

	testIterator = iter(testloader)
	c3 = next(testIterator)
	testData = c3[0].clone().detach()
	testLabels = c3[1].clone().detach()

	return {"data":trainData, "label":trainLabels}, {"data":valData, "label":valLabels}, {"data":testData, "label":testLabels}



			
if __name__ == '__main__':
	options = getOptions(sys.argv[1:])
	L = ["options.ckptDir: " + options.ckptDir,"options.maxDepth: " + str(options.maxDepth),"options.cnnLR: " + str(options.cnnLR),"options.mlpLR: " + str(options.mlpLR),"options.cnnEpochs: " + str(options.cnnEpochs),"options.mlpEpochs: " + str(options.mlpEpochs),"options.cnnOut: " + str(options.cnnOut),"options.mlpFC1: " + str(options.mlpFC1),"options.mlpFC2: " + str(options.mlpFC2),"options.trainFlg: " + str(options.trainFlg)]
	print(L)

	trainInputDict, valInputDict, testInputDict = loadNewDictionaries()
	print("len(trainInputDict[\"data\"]): ",len(trainInputDict["data"]), ",  len(valInputDict[\"data\"]): ",len(valInputDict["data"]), ",  len(testInputDict[\"data\"]): ",len(testInputDict["data"]))		
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	tree = Tree(device, maxDepth=options.maxDepth, classThreshold = 2, dataNumThreshold = 100, numClasses = 10)
	
	start = time.time()
	if options.trainFlg == True:
		resumeFromNodeId = -1
		tree.tree_traversal(trainInputDict, valInputDict, resumeTrain=False, resumeFromNodeId=resumeFromNodeId)

		# resumeFromNodeId = 2
		# tree.tree_traversal(trainInputDict, valInputDict, resumeTrain=True, resumeFromNodeId=resumeFromNodeId)

	tree.testTraversal(testInputDict)

	tree.printTree()

	end = time.time()
	print("Time Taken by whole program is ", float(end-start)/60.0, " minutes.")