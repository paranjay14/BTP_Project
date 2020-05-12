import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn import CNN
# from mlp import MLP
from mlp_small import MLP
from torchsummary import summary
import time
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import smote_variants as sv
import random
import numpy as np
from getOptions import options
import time

def kmeans_output(all_images_flat, device, num_clusters=2):
	cluster_ids_x, cluster_centers = kmeans(X=all_images_flat, num_clusters=num_clusters, distance='euclidean', device=device)
	return cluster_ids_x, cluster_centers


class myNode:
	def __init__(self, parentId, nodeId, device, isTrain, level, parentNode=None):
		self.parentId = parentId
		self.nodeId = nodeId
		self.device = device
		self.isTrain = isTrain
		self.level = level
		self.children = []
		if parentNode:
			parentNode.children.append(self)

	def __str__(self):
		return str(self.nodeId)

	def setInput(self, trainInputDict, valInputDict, numClasses, giniValue, isLeaf, leafClass, lchildId, rchildId):
		self.trainInputDict = trainInputDict
		self.valInputDict = valInputDict
		imgSize = trainInputDict["data"][0].shape[2]
		inChannels = trainInputDict["data"][0].shape[0]
		print("nodeId: ", self.nodeId, ", imgTensorShape : ", trainInputDict["data"].shape)
		outChannels = options.cnnOut
		outChannels = options.cnnOut + 8*(self.level)
		kernel = 5
		self.cnnModel = CNN(img_size=imgSize, in_channels=inChannels, out_channels=outChannels, num_class=numClasses, kernel=kernel, use_bn=False)
		numFeatures = self.cnnModel.features
		self.mlpModel = MLP(numFeatures, use_bn=False)
		self.numClasses = numClasses
		self.giniValue = giniValue
		self.giniGain = 0.0
		self.splitAcc = 0.0
		self.isLeaf = isLeaf
		self.leafClass = leafClass
		self.lchildId = lchildId
		self.rchildId = rchildId
		self.numData = trainInputDict["data"].shape[0]
		self.nodeAcc = [0,0,0.0]
		self.classLabels = { l:cnt for l,cnt in zip(*np.unique(trainInputDict["label"].numpy(),return_counts=True))}
		print("nodeId:", self.nodeId, ",  parentId:", self.parentId, ",  level:", self.level, ",  lchildId:", self.lchildId, ",  rchildId:", self.rchildId, ",  isLeaf:", self.isLeaf, ",  leafClass:", self.leafClass, ",  numClasses:", self.numClasses, ",  numData:", self.numData)

	def trainCNN(self, labelMap, reverseLabelMap):
		loss_fn = nn.CrossEntropyLoss()
		# loss_fn_mse = nn.MSELoss()
		optimizer = torch.optim.Adam(self.cnnModel.parameters(),lr=options.cnnLR)
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, options.cnnSchEpochs, options.cnnSchFactor)
		# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.2, mode='max',verbose=True)
		self.cnnModel.to(self.device)
		trainLabels = self.trainInputDict["label"]
		trainInputs = self.trainInputDict["data"]
		trainLabels = trainLabels.to(self.device)
		trainInputs = trainInputs.to(self.device)

		numBatches = options.cnnBatches
		batchSize = int((len(trainInputs))/(numBatches-1))
		if(batchSize<3):
			batchSize=int(len(trainInputs))
			numBatches=1
# 		numEpochs = max(60, options.cnnEpochs - 10*(self.level))
		numEpochs = options.cnnEpochs - 10*(self.level)
		# numEpochs = options.cnnEpochs

		st_btch = 0
		batch_sep = []
		for i in range(numBatches):
			end_btch = min(st_btch + batchSize, len(trainInputs))
			if end_btch == st_btch:
				numBatches-=1
				break
			else:
				batch_sep.append([st_btch, end_btch])
				st_btch = end_btch

		train_loss = 0
		for epoch in range(numEpochs):
			self.cnnModel.train()
			total = 0
			correct = 0
			train_loss = 0
			random.shuffle(batch_sep)
			for batch in range(numBatches):
				st_btch, end_btch = batch_sep[batch]
				# print("st_btch: ",st_btch, ",  end_btch: ",end_btch, ",  trainLabels.shape: ",trainLabels.shape )
				optimizer.zero_grad()
				_, _, est_labels, feat_same = self.cnnModel(trainInputs[st_btch:end_btch])
				batch_loss_label = loss_fn(est_labels, trainLabels[st_btch:end_btch])
				# print(feat_same.shape)
				# print(trainInputs[st_btch:end_btch].shape)
				# batch_loss_featr = loss_fn_mse(feat_same, trainInputs[st_btch:end_btch])
				# batch_loss = batch_loss_featr + batch_loss_label
				batch_loss =  batch_loss_label
				batch_loss.backward()
				optimizer.step()
				# print(batch_loss_featr.item(), batch_loss_label.item())
				# if batch == 0:
				# 	train_loss_tensor = batch_loss
				# else:
				# 	train_loss_tensor += batch_loss
				train_loss += batch_loss.item()
				_, predicted = est_labels.max(1)
				total += end_btch - st_btch
				correct += predicted.eq(trainLabels[st_btch:end_btch]).sum().item()
			# scheduler.step(train_loss_tensor)
			scheduler.step()
			# train_loss = train_loss_tensor.item()
			#TODO: Add validation iteration here(first change mode to eval)

			if (epoch%options.cnnSchEpochs == options.cnnSchEpochs-1) or (epoch%options.cnnSchEpochs == 0):
				self.cnnModel.eval()
				_, _, est_labels, _ = self.cnnModel(self.valInputDict["data"].to(self.device))
				val_loss = loss_fn(est_labels, self.valInputDict["label"].to(self.device))
				_, predicted = est_labels.max(1)
				valTotalSize = float(len(self.valInputDict["data"]))
				valCorrect = predicted.eq(self.valInputDict["label"].to(self.device)).sum().item()
				# scheduler.step(valCorrect)
				print(epoch, 'Train Loss: %.3f | Train Acc: %.3f | Val Loss: %.3f | Val Accuracy: %.3f'% (train_loss, 100.*correct/total, val_loss.item(), 100.*float(valCorrect)/valTotalSize))
# 			print(epoch, 'Train Loss: %.3f | Train Acc: %.3f '% (train_loss, 100.*correct/total))

		epoch = numEpochs
		torch.save({
				'epoch':epoch,
				'model_state_dict':self.cnnModel.state_dict(),
				'optimizer_state_dict':optimizer.state_dict(),
				'train_loss':train_loss,
				'labelMap':labelMap,
				'reverseLabelMap':reverseLabelMap,  
				}, options.ckptDir+'/node_cnn_'+str(self.nodeId)+'.pth')


	def trainMLP(self, trainInputs, trainTargets, weightVector):
		# loss_fn = nn.CrossEntropyLoss()
		loss_fn = nn.BCELoss(reduction='none')
		optimizer = torch.optim.Adam(self.mlpModel.parameters(),lr=options.mlpLR)
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, options.mlpSchEpochs, options.mlpSchFactor)

		trainInputs = trainInputs.to(self.device)
		trainTargets = trainTargets.to(self.device)
		weightVector = weightVector.to(self.device)
		self.mlpModel.to(self.device)

		numBatches = options.mlpBatches
		batchSize = int((len(trainInputs))/(numBatches-1))
		if(batchSize<3):
			batchSize=int(len(trainInputs))
			numBatches=1

		st_btch = 0
		batch_sep = []

		for i in range(numBatches):
			end_btch = min(st_btch + batchSize, len(trainInputs))
			if end_btch == st_btch:
				numBatches-=1
				break
			else:
				batch_sep.append([st_btch, end_btch])
				st_btch = end_btch

		numEpochs = max(30, options.mlpEpochs-10*(self.level))
		self.mlpModel.train()
		train_loss=0
		for epoch in range(numEpochs):
			train_loss = 0
			correct = 0
			total = 0
			random.shuffle(batch_sep)

			for batch in range(numBatches):    
				st_btch, end_btch = batch_sep[batch]
				# print("st_btch: ",st_btch, ",  end_btch: ",end_btch, ",  trainTargets.shape: ",trainTargets.shape )
				optimizer.zero_grad()
				est_labels = self.mlpModel(trainInputs[st_btch:end_btch])
				# print(est_labels.shape)
				est_labels = est_labels.view(-1)
				# batch_loss = loss_fn(est_labels, trainTargets[st_btch:end_btch]) #if crossentropy
				# print(est_labels.shape)
				# print(weightVector[st_btch:end_btch].shape)
				# print(trainTargets[st_btch:end_btch].shape)
				batch_loss = weightVector[st_btch:end_btch] * loss_fn(est_labels, trainTargets[st_btch:end_btch].float()) #if bce
				batch_loss = batch_loss.mean()
				batch_loss.backward()
				optimizer.step()

				train_loss += batch_loss.item()
				# _, predicted = est_labels.max(1) #if cross entropy
				predicted = est_labels.detach() #if bce
				predicted += 0.5
				predicted = predicted.long()
				# total += trainTargets.size(0)
				total += end_btch - st_btch
				correct += predicted.eq(trainTargets[st_btch:end_btch]).sum().item()
			scheduler.step()
			#TODO: add validation testing of MLP here
			
			print(epoch, 'Loss: %.3f | Acc: %.3f'% (train_loss, 100.*correct/total))

		epoch = numEpochs
		torch.save({
				'epoch':epoch,
				'model_state_dict':self.mlpModel.state_dict(),
				'optimizer_state_dict':optimizer.state_dict(),
				'train_loss':train_loss,
				}, options.ckptDir+'/node_mlp_'+str(self.nodeId)+'.pth')


	def balanceData(self):
		shape=self.trainInputDict["data"].shape
		print("trainInputDict[data].shape : ", shape)
		copy = self.trainInputDict["data"]
		copy = copy.reshape(shape[0], -1)
		print("copy.shape : ", copy.shape)
		npDict = copy.numpy()
		copyLabel = self.trainInputDict["label"]
		print("copyLabel.shape : ", copyLabel.shape)
		# copyLabel = copyLabel.view(-1)
		npLabel = copyLabel.numpy()
		# [print('Class {} had {} instances originally'.format(label, count)) for label, count in zip(*np.unique(npLabel, return_counts=True))]
		# X_resampled, y_resampled = kmeans_smote.fit_sample(npDict, npLabel)

		# print(sv.get_all_oversamplers_multiclass())

		oversampler= sv.MulticlassOversampling(sv.SMOTE(n_jobs=6))

		X_resampled, y_resampled = oversampler.sample(npDict, npLabel)
		[print('Class {} has {} instances after oversampling'.format(label, count)) for label, count in zip(*np.unique(y_resampled, return_counts=True))]

		newData = torch.from_numpy(X_resampled.reshape(len(X_resampled), shape[1], shape[2], shape[3]))
		newLabel = torch.from_numpy(y_resampled)
		newData = newData.float()
		return newData, newLabel

	def make_labels_list(self):
		labelsList = []
		labelMap = {}
		reverseLabelMap = {}
		for i in self.trainInputDict["label"]:
			if (i.item() not in labelsList):
				labelsList.append(i.item())
			if len(labelsList) == self.numClasses:
				break

		for i, val in enumerate(sorted(labelsList)):
			labelMap[val] = i
			reverseLabelMap[i] = val

		for i, val in enumerate(self.trainInputDict["label"]):
			self.trainInputDict["label"][i] = labelMap[val.item()]

		for i, val in enumerate(self.valInputDict["label"]):
			self.valInputDict["label"][i] = labelMap[val.item()]

		# newData, newLabel = self.balanceData();
		# self.trainInputDict["data"] = newData
		# self.trainInputDict["label"] = newLabel
		return labelMap, reverseLabelMap

	def loadMLPModel(self):
		ckpt = torch.load(options.ckptDir+'/node_mlp_'+str(self.nodeId)+'.pth')
		self.mlpModel.load_state_dict(ckpt['model_state_dict'])
		self.mlpModel.eval()
		self.mlpModel.to(self.device)

	def loadCNNModel(self):
		ckpt = torch.load(options.ckptDir+'/node_cnn_'+str(self.nodeId)+'.pth')
		self.cnnModel.load_state_dict(ckpt['model_state_dict'])
		self.cnnModel.eval()
		self.cnnModel.to(self.device)
		return ckpt['reverseLabelMap'], ckpt['labelMap']

	def separateLabels(self, cluster_ids):
		leftCnt = {}
		rightCnt = {}
		for i in range(len(self.trainInputDict["data"])):
			label = self.trainInputDict["label"][i].item() # 0-indexed according to self.trainInputDict["label"]
			if cluster_ids[i] == 0:
				if label in leftCnt:
					leftCnt[label]+=1
				else:
					leftCnt[label] = 1
			else:
				if label in rightCnt:
					rightCnt[label]+=1
				else:
					rightCnt[label] = 1
		expected_dict = {}
		
		lCnt=0
		rCnt=0
		maxLeftRatio=0.0
		maxRightRatio=0.0
		lIndx=-1
		rIndx=-1

		for label, count in leftCnt.items():
			lRatio=1.0
			if label in rightCnt:
				lRatio = float(count)/float(count+rightCnt[label])
				if lRatio>maxLeftRatio:
					maxLeftRatio = lRatio
					lIndx = label
				# rRatio = float(rightCnt[label])/float(count+rightCnt[label])
				rRatio = 1.0 - lRatio
				if rRatio>maxRightRatio:
					maxRightRatio = rRatio
					rIndx = label
				
				if count >= rightCnt[label]:
					expected_dict[label] = 0
					lCnt+=1
				else:
					expected_dict[label] = 1
					rCnt+=1
			else:
				if lRatio>maxLeftRatio:
					maxLeftRatio = lRatio
					lIndx = label
				expected_dict[label] = 0
				lCnt+=1

		for label, count in rightCnt.items():
			if not (label in expected_dict):
				rRatio=1.0
				if rRatio>maxRightRatio:
					maxRightRatio = rRatio
					rIndx = label
				expected_dict[label] = 1
				rCnt+=1

		if lCnt==0:
			expected_dict[lIndx] = 0
		if rCnt==0:
			expected_dict[rIndx] = 1

		print("printing expected split from k means")
		print(expected_dict)                
		leftSortedListOfTuples = sorted(leftCnt.items(), reverse=True, key=lambda x: x[1])
		rightSortedListOfTuples = sorted(rightCnt.items(), reverse=True, key=lambda x: x[1])
		return leftSortedListOfTuples, rightSortedListOfTuples, expected_dict

	def makeFinalDict(self, leftSortedListOfTuples, rightSortedListOfTuples, expected_dict):
		final_dict = expected_dict

		# final_dict = {}
		# fullDict = {}

		# for ind, element in enumerate(leftSortedListOfTuples):
		# 	fullDict[element[0]] = element[1]
		# 	if ind >= self.numClasses/2:
		# 		final_dict[element[0]] = 1
		# 	else:
		# 		final_dict[element[0]] = 0  

		# for ind, element in enumerate(rightSortedListOfTuples):
		# 	if element[0] not in fullDict:
		# 		fullDict[element[0]] = -1*element[1]

		# 	if ind >= self.numClasses/2:
		# 		final_dict[element[0]] = 0
		# 	else:
		# 		final_dict[element[0]] = 1        

		# fullSortedListOfTuples = sorted(fullDict.items(), reverse=True, key=lambda x: (x[1],x[0]))

		## if (self.numClasses%2 == 1):
		## 	final_dict[fullSortedListOfTuples[int(self.numClasses/2)][0]] = -1

		print("Printing final_dict items...")
		print(final_dict)

		#TODO: separate for validation set too
		torch.save({
				'splittingDict':final_dict,
				}, options.ckptDir+'/node_split_'+str(self.nodeId)+'.pth')
		return final_dict

	def getSavedFinalSplit(self):
		ckpt = torch.load(options.ckptDir+'/node_split_'+str(self.nodeId)+'.pth')
		return ckpt['splittingDict']

	def setFinalPredictions(self, predicted):
		ckpt = torch.load(options.ckptDir+'/testPred.pth')
		testPredDict = ckpt['testPredDict']
		testPredDict['actual'] = testPredDict['actual'].to(self.device)
		testPredDict['pred'] = testPredDict['pred'].to(self.device)
		testPredDict['actual'] = torch.cat((testPredDict['actual'],self.trainInputDict["label"].to(self.device)),0)
		testPredDict['pred'] = torch.cat((testPredDict['pred'],predicted),0)
		torch.save({
			'testPredDict':testPredDict,
			}, options.ckptDir+'/testPred.pth')

	def checkTestPreds(self, reverseLabelMap, est_labels):
		_, predicted = est_labels.max(1)
		predicted = predicted.to(self.device)
		for i, val in enumerate(predicted):
			predicted[i] = reverseLabelMap[val.item()]

		correct = predicted.eq(self.trainInputDict["label"].to(self.device)).sum().item()
		total = len(est_labels)
		self.nodeAcc[2] = round(100.*correct/total,3)
		self.nodeAcc[1] = total
		self.nodeAcc[0] = correct
		print('Node %d Acc: %.3f'% ( self.nodeId, self.nodeAcc[2]))
		
		NodeDict = torch.load(options.ckptDir+'/node_'+str(self.nodeId)+'.pth')['nodeDict']
		NodeDict['nodeAcc'] = self.nodeAcc[:]
		torch.save({
			'nodeDict':NodeDict,
			}, options.ckptDir+'/node_'+str(self.nodeId)+'.pth')

		LevelDict = torch.load(options.ckptDir+'/level.pth')['levelDict']
		LevelDict['levelAcc'][self.level][0] += correct
		LevelDict['levelAcc'][self.level][1] += total
		if(self.isLeaf):
			LevelDict['leafAcc'][0] += correct
			LevelDict['leafAcc'][1] += total
		torch.save({
			'levelDict':LevelDict,
			}, options.ckptDir+'/level.pth')

		if self.isLeaf:
			self.setFinalPredictions(predicted)

	def doLabelCounting(self, mlpPrediction):
		lclasses = [0]*10 # 0-indexed according to self.trainInputDict["label"]
		rclasses = [0]*10

		for i, val in enumerate(mlpPrediction):
			if val<=0.5:
				lclasses[self.trainInputDict["label"][i].item()]+=1
			else:
				rclasses[self.trainInputDict["label"][i].item()]+=1
		return lclasses, rclasses

	def countBalanceAndThreshold(self, mlpPrediction, labelMap):
		final_dict = self.getSavedFinalSplit()
		lclasses, rclasses = self.doLabelCounting(mlpPrediction)
		# totalLeftImages = 0.0
		# totalRightImages = 0.0
		maxLeftClasses = 0.0
		maxRightClasses = 0.0
		testCorrectResults = 0.0
		totalL = 0
		totalR = 0
		# print(final_dict)
		for i, val in enumerate(lclasses):
			# totalLeftImages += val
			if not self.isTrain and (i in labelMap):
				totalL+=val
				if(final_dict[labelMap[i]] == 0 or final_dict[labelMap[i]] == -1):
					testCorrectResults += val
			maxLeftClasses = max(maxLeftClasses, val)

		for i, val in enumerate(rclasses):
			# totalRightImages += val
			if not self.isTrain and (i in labelMap):
				totalR+=val
				if (final_dict[labelMap[i]] == 1 or final_dict[labelMap[i]] == -1):
					testCorrectResults += val
			maxRightClasses = max(maxRightClasses, val)

		if not self.isTrain:
			# total = float(len(self.trainInputDict["label"]))
			total = float(totalL+totalR)
			splitAcc = float(100.*testCorrectResults/total)
			print('Split Acc: %.3f'% (splitAcc))
			NodeDict = torch.load(options.ckptDir+'/node_'+str(self.nodeId)+'.pth')['nodeDict']
			NodeDict['splitAcc'] = splitAcc
			torch.save({
				'nodeDict':NodeDict,
				}, options.ckptDir+'/node_'+str(self.nodeId)+'.pth')

		leftClassesToBeRemoved = []
		rightClassesToBeRemoved = []

		# threshold = 15.0
		# for i, val in enumerate(lclasses):
		# 	if float(100*val)/maxLeftClasses < threshold:
		# 		leftClassesToBeRemoved.append(i)

		# for i, val in enumerate(rclasses):
		# 	if float(100*val)/maxRightClasses < threshold:
		# 		rightClassesToBeRemoved.append(i)
		
		return maxLeftClasses, maxRightClasses, testCorrectResults, leftClassesToBeRemoved, rightClassesToBeRemoved

	def getPredictionAnalysis(self, totalLeftImages, totalRightImages, lclasses, rclasses):
		giniLeftRatio = 0.0
		giniRightRatio = 0.0
		lcheck = 0.0
		rcheck = 0.0
		print("# of Left images: ", totalLeftImages)
		print("# of Right images: ", totalRightImages)
		noOfLeftClasses = 0
		noOfRightClasses = 0
		for i in lclasses:
			if i != 0:
				noOfLeftClasses += 1
			pi = float(i)/totalLeftImages
			lcheck += pi
			giniLeftRatio += pi*(1-pi)

		# print("---")
		for i in rclasses:
			if i != 0:
				noOfRightClasses += 1
			pi = float(i)/totalRightImages
			rcheck += pi
			giniRightRatio += pi*(1-pi)

		print("giniRightRatio: ", giniRightRatio)
		print("giniLeftRatio: ", giniLeftRatio)

		leftChildrenRatio = totalLeftImages/totalRightImages

		impurityDrop = leftChildrenRatio*float(giniLeftRatio) + (1-leftChildrenRatio)*float(giniRightRatio)
		giniGain = self.giniValue - impurityDrop

		print("impurityDrop: ", impurityDrop)
		print("giniGain: ", giniGain)
		print("lclasses: ", lclasses)
		print("rclasses: ", rclasses)
		print("noOfLeftClasses: ", noOfLeftClasses)
		print("noOfRightClasses: ", noOfRightClasses)
		return giniLeftRatio, giniRightRatio, noOfLeftClasses, noOfRightClasses, giniGain
		


	def classifyLabels(self, mlpPrediction, reverseLabelMap, labelMap):
		maxLeftClasses, maxRightClasses, testCorrectResults, leftClassesToBeRemoved, rightClassesToBeRemoved = self.countBalanceAndThreshold(mlpPrediction, labelMap)
		totalLeftImages = 0.0
		totalRightImages = 0.0
		trainLimages = []
		trainRimages = []
		trainLLabels = []
		trainRLabels = []
		leftChildIndexList = []
		rightChildIndexList = []
		leftValIndexList = []
		rightValIndexList = []
		lclasses = [0]*10 # 0-indexed in Train according to self.trainInputDict["label"]
		rclasses = [0]*10
		maxLeft=0
		maxRight=0
		maxLeftClassIndex=-1
		maxRightClassIndex=-1
		final_dict = self.getSavedFinalSplit()
		splitClassIndx = -1
		splitClassCnt = -1
		leftSplitClassCnt = 0
		rightSplitClassCnt = 0
		lmaxSplitClassCnt = 0
		rmaxSplitClassCnt = 0
		# resDict = dict(zip(self.trainInputDict["label"].tolist(), mlpPrediction.tolist()))
		testLclasses = [0]*10
		testRclasses = [0]*10


		for k,v in final_dict.items():
			if v == -1:
				splitClassIndx = k
				splitClassCnt = (self.trainInputDict["label"].tolist()).count(splitClassIndx)
				lmaxSplitClassCnt = int((splitClassCnt+1)/2)
				rmaxSplitClassCnt = int(splitClassCnt/2)
				break


		for i, val in enumerate(mlpPrediction):
			label = self.trainInputDict["label"][i].item()
			# print("i : ", i)
			if val<=0.5:
				if self.isTrain:
					### if not (label in leftClassesToBeRemoved):
					if (final_dict[label] == 0) or ((final_dict[label] == -1) and leftSplitClassCnt<lmaxSplitClassCnt):
						# trainLimages.append((image_next[i].detach()).tolist())
						lclasses[label]+=1
						# testLclasses[reverseLabelMap[label]]+=1
						# trainLLabels.append(reverseLabelMap[label])
						leftChildIndexList.append(i)
						if (final_dict[label] == -1):
							leftSplitClassCnt += 1
						if lclasses[label] > maxLeft:
							maxLeft = lclasses[label]
							maxLeftClassIndex = reverseLabelMap[label]
					else:
						# trainRimages.append((image_next[i].detach()).tolist())
						rclasses[label]+=1
						# testLclasses[reverseLabelMap[label]]+=1
						# trainRLabels.append(reverseLabelMap[label])
						rightChildIndexList.append(i)
						if (final_dict[label] == -1):
							rightSplitClassCnt += 1
						if rclasses[label] > maxRight:
							maxRight = rclasses[label]
							maxRightClassIndex = reverseLabelMap[label]
					self.trainInputDict["label"][i] = reverseLabelMap[self.trainInputDict["label"][i].item()]
				else:
					# trainLimages.append((image_next[i].detach()).tolist())
					lclasses[label]+=1
					# testLclasses[reverseLabelMap[label]]+=1
					# trainLLabels.append(label)
					leftChildIndexList.append(i)

			else:
				if self.isTrain:
					### if not (label in rightClassesToBeRemoved):
					if (final_dict[label] == 1) or ((final_dict[label] == -1) and leftSplitClassCnt>=lmaxSplitClassCnt):
						# trainRimages.append((image_next[i].detach()).tolist())
						rclasses[label]+=1
						# testRclasses[reverseLabelMap[label]]+=1
						# trainRLabels.append(reverseLabelMap[label])
						rightChildIndexList.append(i)
						if (final_dict[label] == -1):
							rightSplitClassCnt += 1
						if rclasses[label] > maxRight:
							maxRight = rclasses[label]
							maxRightClassIndex = reverseLabelMap[label]
					else:
						# trainLimages.append((image_next[i].detach()).tolist())
						lclasses[label]+=1
						# testRclasses[reverseLabelMap[label]]+=1
						# trainLLabels.append(reverseLabelMap[label])
						leftChildIndexList.append(i)
						if (final_dict[label] == -1):
							leftSplitClassCnt += 1
						if lclasses[label] > maxLeft:
							maxLeft = lclasses[label]
							maxLeftClassIndex = reverseLabelMap[label]
					self.trainInputDict["label"][i] = reverseLabelMap[self.trainInputDict["label"][i].item()]
				else:
					# trainRimages.append((image_next[i].detach()).tolist())
					rclasses[label]+=1
					# trainRLabels.append(label)
					# testRclasses[reverseLabelMap[label]]+=1
					rightChildIndexList.append(i)

		totalLeftImages += float(len(leftChildIndexList))
		totalRightImages += float(len(rightChildIndexList))

		# print(testLclasses)
		# print(testRclasses)

		# lTrainDict = {"data":torch.tensor(trainLimages), "label":torch.tensor(trainLLabels)}
		# rTrainDict = {"data":torch.tensor(trainRimages), "label":torch.tensor(trainRLabels)}
		lTrainDict = {"data":self.trainInputDict["data"][leftChildIndexList], "label":self.trainInputDict["label"][leftChildIndexList]}
		rTrainDict = {"data":self.trainInputDict["data"][rightChildIndexList], "label":self.trainInputDict["label"][rightChildIndexList]}
		print("lTrainDict[data].shape: ", lTrainDict["data"].shape, "  lTrainDict[label].shape: ", lTrainDict["label"].shape)
		print("rTrainDict[data].shape: ", rTrainDict["data"].shape, "  rTrainDict[label].shape: ", rTrainDict["label"].shape)

		lValDict={}
		rValDict={}

		if self.isTrain:
			for i, val in enumerate(self.valInputDict["label"]):
				label = self.valInputDict["label"][i].item()
				if (final_dict[label] == 0):
					leftValIndexList.append(i)
				elif (final_dict[label] == 1):
					rightValIndexList.append(i)
				self.valInputDict["label"][i] = reverseLabelMap[val.item()]

			lValDict = {"data":self.valInputDict["data"][leftValIndexList], "label":self.valInputDict["label"][leftValIndexList]}
			rValDict = {"data":self.valInputDict["data"][rightValIndexList], "label":self.valInputDict["label"][rightValIndexList]}
			print("lValDict[data].shape: ", lValDict["data"].shape, "  lValDict[label].shape: ", lValDict["label"].shape)
			print("rValDict[data].shape: ", rValDict["data"].shape, "  rValDict[label].shape: ", rValDict["label"].shape)

		giniLeftRatio, giniRightRatio, noOfLeftClasses, noOfRightClasses, giniGain = self.getPredictionAnalysis(totalLeftImages, totalRightImages, lclasses, rclasses)

		leftDataNum = int(totalLeftImages)
		rightDataNum = int(totalRightImages)

		if not (len(leftChildIndexList) == 0):
			maxLeft = float(float(maxLeft)/float(len(leftChildIndexList)))
		if not (len(rightChildIndexList) == 0):
			maxRight = float(float(maxRight)/float(len(rightChildIndexList)))

		handleLeafDict = {"lvl":self.level+1,"noOfLeftClasses":noOfLeftClasses, "noOfRightClasses":noOfRightClasses, "maxLeft":maxLeft, "maxRight":maxRight, "leftDataNum":leftDataNum, "rightDataNum":rightDataNum,"maxLeftClassIndex":maxLeftClassIndex,"maxRightClassIndex":maxRightClassIndex, "giniGain":giniGain}		


		print("RETURNING FROM WORK...")

		if self.isTrain and not self.isLeaf:
			return lTrainDict, lValDict, rTrainDict, rValDict, giniLeftRatio, giniRightRatio, handleLeafDict
		elif not self.isTrain and not self.isLeaf:
			return lTrainDict, rTrainDict, giniLeftRatio, giniRightRatio, noOfLeftClasses, noOfRightClasses
		elif self.isTrain and self.isLeaf:
			return
		else:
			return


	def workTest(self):
		if not (self.leafClass == -1):
			x=torch.Tensor(1,1).long()
			x[0] = 1
			est_labels = torch.cat(len(self.trainInputDict["label"].to(self.device))*[x])
			reverseLabelMap = {}
			reverseLabelMap[0] = self.leafClass
			self.checkTestPreds(reverseLabelMap, est_labels)
			return

		else:
			reverseLabelMap, labelMap = self.loadCNNModel()
			
			trainInputs = self.trainInputDict["data"]
			numBatches = 4
			batchSize = int((len(trainInputs))/(numBatches-1))
			if(batchSize<3):
				batchSize=int(len(trainInputs))
				numBatches=1

			st_btch = 0
			batch_sep = []

			for i in range(numBatches):
				end_btch = min(st_btch + batchSize, len(trainInputs))
				if end_btch == st_btch:
					numBatches-=1
					break
				else:
					batch_sep.append([st_btch, end_btch])
					st_btch = end_btch

			imgNextFlat=[]
			estLabels=[]
			# imgNext=[]

			for batch in range(numBatches):
				st_btch, end_btch = batch_sep[batch]
				_, image_next_flat, est_labels, _ = self.cnnModel(self.trainInputDict["data"][st_btch:end_btch].to(self.device))
				if batch == 0:
					imgNextFlat = image_next_flat.detach().cpu()	
					estLabels = est_labels.detach().cpu()
					# imgNext = image_next
				else:
					imgNextFlat = torch.cat((imgNextFlat,image_next_flat.detach().cpu()))
					estLabels = torch.cat((estLabels,est_labels.detach().cpu()))
					# imgNext = torch.cat((imgNext,image_next))


			# _, image_next_flat, est_labels, _ = self.cnnModel(self.trainInputDict["data"].to(self.device))
			# imgNextFlat = image_next_flat.detach().cpu()	
			# estLabels = est_labels.detach().cpu()

			if not self.isTrain:
				self.checkTestPreds(reverseLabelMap, estLabels.to(self.device))
				
			if self.isLeaf:
				return

			self.loadMLPModel()

			estLabels = []
			for batch in range(numBatches):    
				st_btch, end_btch = batch_sep[batch]
				est_labels = self.mlpModel(imgNextFlat[st_btch:end_btch].to(self.device))
				if batch == 0:
					estLabels = est_labels.detach().cpu()
				else:
					estLabels = torch.cat((estLabels,est_labels.detach().cpu()))

			# estLabels = self.mlpModel(imgNextFlat.to(self.device))

			estLabels = estLabels.view(-1)
			mlpPrediction = estLabels.detach()
			mlpPrediction += 0.5
			mlpPrediction = mlpPrediction.long()
			return self.classifyLabels(mlpPrediction, reverseLabelMap, labelMap)

	
	def getTrainPredictionsNotLeaf(self):
		self.loadCNNModel()

		trainInputs = self.trainInputDict["data"]
		numBatches = 3
		batchSize = int((len(trainInputs))/(numBatches-1))
		if(batchSize<3):
			batchSize=int(len(trainInputs))
			numBatches=1

		st_btch = 0
		batch_sep = []

		for i in range(numBatches):
			end_btch = min(st_btch + batchSize, len(trainInputs))
			if end_btch == st_btch:
				numBatches-=1
				break
			else:
				batch_sep.append([st_btch, end_btch])
				st_btch = end_btch

		imgNextFlat = []

		for batch in range(numBatches):
			st_btch, end_btch = batch_sep[batch]
			_, image_next_flat, _, _  = self.cnnModel(self.trainInputDict["data"][st_btch:end_btch].to(self.device))
			if batch == 0:
				imgNextFlat = image_next_flat.detach().cpu()
			else:
				imgNextFlat = torch.cat((imgNextFlat,image_next_flat.detach().cpu()))

		# _, image_next_flat, _, _  = self.cnnModel(self.trainInputDict["data"].to(self.device))
		# imgNextFlat = image_next_flat.detach().cpu()

		print("image_next_flat.shape : ", imgNextFlat.shape)
		img_flat_nmpy = imgNextFlat.numpy()
		countImageTotal = imgNextFlat.shape[0]

		start = time.time()
		# kmeans = KMeans(n_clusters=2, n_jobs=-1).fit(img_flat_nmpy)
		kmeans = MiniBatchKMeans(n_clusters=2,random_state=0).fit(img_flat_nmpy)
		end = time.time()
		print("Time Taken by Kmeans is ", end-start)

		cluster_ids = kmeans.labels_
		print("Kmeans completed successfully...")		


		leftSortedListOfTuples, rightSortedListOfTuples, expected_dict = self.separateLabels(cluster_ids)
		final_dict = self.makeFinalDict(leftSortedListOfTuples, rightSortedListOfTuples, expected_dict)

		expectedMlpLabels = []
		weightVector = []
		countImageRight = 0
		countImageLeft = 0

		# countImageRight = int(np.sum(cluster_ids))
		# countImageLeft = countImageTotal - countImageRight
		leftSplitClassCnt = 0
		rightSplitClassCnt = 0
		lmaxSplitClassCnt = 0
		rmaxSplitClassCnt = 0

		for label, count in zip(*np.unique(self.trainInputDict['label'], return_counts=True)):
			if final_dict[label] == 0:
				countImageLeft += count
			elif final_dict[label] == 1:
				countImageRight += count
			else:
				countImageLeft += int((count+1)/2)
				countImageRight += int((count)/2)
				lmaxSplitClassCnt = int((count+1)/2)
				rmaxSplitClassCnt = int(count/2)


		print("Image Statistics before MLP : L R : ", countImageLeft, countImageRight)
		minCountImage = min(countImageLeft, countImageRight)
		for i in range(len(self.trainInputDict["data"])):
			label = self.trainInputDict["label"][i].item()
			
			# expectedMlpLabels.append(cluster_ids[i])
			# if cluster_ids[i] == 0:

			if (final_dict[label] == 0) or ((final_dict[label] == -1) and (leftSplitClassCnt<lmaxSplitClassCnt)):
				expectedMlpLabels.append(0)
				weightVector.append(minCountImage/countImageLeft)
				if (final_dict[label] == -1):
					leftSplitClassCnt += 1
			else:
				expectedMlpLabels.append(1)
				weightVector.append(minCountImage/countImageRight)
				if (final_dict[label] == -1):
					rightSplitClassCnt += 1

		expectedMlpLabels = torch.tensor(expectedMlpLabels).cpu()
		weightVector = torch.tensor(weightVector, dtype=torch.float32).cpu()
		# print(weightVector)
		# print(expectedMlpLabels)
		print("expectedMlpLabels.shape : ",expectedMlpLabels.shape)
		return imgNextFlat, expectedMlpLabels, weightVector


	def workTrain(self):
		nodeDict = {}
		nodeDict['nodeId'] = self.nodeId
		nodeDict['parentId'] = self.parentId
		nodeDict['level'] = self.level
		nodeDict['isLeaf'] = self.isLeaf
		nodeDict['leafClass'] = self.leafClass
		nodeDict['lchildId'] = self.lchildId
		nodeDict['rchildId'] = self.rchildId
		nodeDict['numClasses'] = self.numClasses
		nodeDict['numData'] = self.numData
		nodeDict['classLabels'] = self.classLabels
		nodeDict['giniGain'] = self.giniGain
		nodeDict['splitAcc'] = self.splitAcc
		nodeDict['nodeAcc'] = self.nodeAcc

		torch.save({
					'nodeDict':nodeDict,
					}, options.ckptDir+'/node_'+str(self.nodeId)+'.pth')

		if not (self.leafClass == -1):
			return
		else:
			oldData = self.trainInputDict["data"]
			oldLabel = self.trainInputDict["label"]
			labelMap, reverseLabelMap = self.make_labels_list()			

			self.cnnModel.to(self.device)
			self.trainCNN(labelMap, reverseLabelMap)
			print("CNN trained successfully...")		

			if not self.isLeaf:
				image_next_flat, expectedMlpLabels, weightVector = self.getTrainPredictionsNotLeaf()
				self.mlpModel.to(self.device)
				self.trainMLP(image_next_flat, expectedMlpLabels, weightVector)
				print("MLP trained successfully...")
			self.trainInputDict["data"] = oldData
			self.trainInputDict["label"] = oldLabel

			if self.isLeaf:
				return
			else:
				return self.workTest()
