import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn import CNN
from mlp_small import MLP
from torchsummary import summary
import time
from sklearn.cluster import KMeans
import numpy as np
import smote_variants as sv
import random


def kmeans_output(all_images_flat, device, num_clusters=2):
	cluster_ids_x, cluster_centers = kmeans(X=all_images_flat, num_clusters=num_clusters, distance='euclidean', device=device)
	return cluster_ids_x, cluster_centers


class Node:
	def __init__(self, parentId, nodeId, device, isTrain, level):
		self.parentId = parentId
		self.nodeId = nodeId
		self.device = device
		self.isTrain = isTrain
		self.level = level

	def setInput(self, trainInputDict, valInputDict, numClasses, giniValue, isLeaf):
		self.trainInputDict = trainInputDict
		self.valInputDict = valInputDict
		imgSize = trainInputDict["data"][0].shape[2]
		inChannels = trainInputDict["data"][0].shape[0]
		print("nodeId: ", self.nodeId, ", imgSize : ", imgSize)
		outChannels = 16
		kernel = 5
		self.cnnModel = CNN(img_size=imgSize, in_channels=inChannels, out_channels=outChannels, num_class=numClasses, kernel=kernel)
		numFeatures = self.cnnModel.features
		self.mlpModel = MLP(numFeatures)
		self.numClasses = numClasses
		self.giniValue = giniValue
		self.isLeaf = isLeaf

	def trainCNN(self, labelMap, reverseLabelMap):
		loss_fn = nn.CrossEntropyLoss()
		loss_fn_mse = nn.MSELoss()
		optimizer = torch.optim.Adam(self.cnnModel.parameters(),lr=0.001)
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, 0.4)
		self.cnnModel.to(self.device)
		trainLabels = self.trainInputDict["label"]
		trainInputs = self.trainInputDict["data"]
		trainLabels = trainLabels.to(self.device)
		trainInputs = trainInputs.to(self.device)

		numBatches = 100
		batchSize = int((len(trainInputs) + numBatches - 1)/numBatches)
		numEpochs = 100

		st_btch = 0
		batch_sep = []
		for i in range(numBatches):
			end_btch = min(st_btch + batchSize, len(trainInputs))
			batch_sep.append([st_btch, end_btch])
			st_btch = end_btch

		self.cnnModel.train()
		for epoch in range(numEpochs):
			total = 0
			correct = 0
			train_loss = 0
			random.shuffle(batch_sep)
			for batch in range(numBatches):
				st_btch, end_btch = batch_sep[batch]
				optimizer.zero_grad()
				_, _, est_labels, feat_same = self.cnnModel(trainInputs[st_btch:end_btch])
				batch_loss_label = loss_fn(est_labels, trainLabels[st_btch:end_btch])
				# print(feat_same.shape)
				# print(trainInputs[st_btch:end_btch].shape)
				batch_loss_featr = loss_fn_mse(feat_same, trainInputs[st_btch:end_btch])
				batch_loss = batch_loss_featr + batch_loss_label
				batch_loss.backward()
				optimizer.step()
				# print(batch_loss_featr.item(), batch_loss_label.item())
				train_loss += batch_loss.item()
				_, predicted = est_labels.max(1)
				total += end_btch - st_btch
				correct += predicted.eq(trainLabels[st_btch:end_btch]).sum().item()
			scheduler.step()

			#TODO: Add validation iteration here(first change mode to eval)
			
			print(epoch, 'Train Loss: %.3f | Train Acc: %.3f'% (train_loss, 100.*correct/total))

			torch.save({
					'epoch':epoch,
					'model_state_dict':self.cnnModel.state_dict(),
					'optimizer_state_dict':optimizer.state_dict(),
					'train_loss':train_loss,
					'labelMap':labelMap,
					'reverseLabelMap':reverseLabelMap,  
					}, 'ckpt/node_cnn_'+str(self.parentId)+'_'+str(self.nodeId)+'.pth')


	def trainMLP(self, trainInputs, trainTargets):
		# loss_fn = nn.CrossEntropyLoss()
		loss_fn = nn.BCELoss()
		optimizer = torch.optim.Adam(self.mlpModel.parameters(),lr=0.001)
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.4)

		trainInputs = trainInputs.to(self.device)
		trainTargets = trainTargets.to(self.device)
		self.mlpModel.to(self.device)

		numBatches = 50
		batchSize = int((len(trainInputs) + numBatches - 1)/numBatches)

		st_btch = 0
		batch_sep = []
		for i in range(numBatches):
			end_btch = min(st_btch + batchSize, len(trainInputs))
			batch_sep.append([st_btch, end_btch])
			st_btch = end_btch

		numEpochs = 60
		self.mlpModel.train()
		for epoch in range(numEpochs):
			train_loss = 0
			correct = 0
			total = 0
			random.shuffle(batch_sep)

			for batch in range(numBatches):    
				st_btch, end_btch = batch_sep[batch]
				optimizer.zero_grad()
				est_labels = self.mlpModel(trainInputs[st_btch:end_btch])
				# print(est_labels.shape)
				est_labels = est_labels.view(-1)
				# batch_loss = loss_fn(est_labels, trainTargets[st_btch:end_btch]) #if crossentropy
				batch_loss = loss_fn(est_labels, trainTargets[st_btch:end_btch].float()) #if bce
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

			torch.save({
					'epoch':epoch,
					'model_state_dict':self.mlpModel.state_dict(),
					'optimizer_state_dict':optimizer.state_dict(),
					'train_loss':train_loss,
					}, 'ckpt/node_mlp_'+str(self.parentId)+'_'+str(self.nodeId)+'.pth')


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

		# oversampler = sv.SMOTE(n_jobs=8)
		X_resampled, y_resampled = oversampler.sample(npDict, npLabel)
		[print('Class {} has {} instances after oversampling'.format(label, count)) for label, count in zip(*np.unique(y_resampled, return_counts=True))]

		newData = torch.from_numpy(X_resampled.reshape(len(X_resampled), shape[1], shape[2], shape[3]))
		newLabel = torch.from_numpy(y_resampled)
		newData = newData.float()
		return newData, newLabel
		# self.trainInputDict["data"] = torch.from_numpy(X_resampled.reshape(len(X_resampled), shape[1], shape[2], shape[3]))
		# self.trainInputDict["label"] = torch.from_numpy(y_resampled)
		# self.trainInputDict["data"] = self.trainInputDict["data"].float()
		# print("trainInputDict[data].shape : ", self.trainInputDict["data"].shape, " trainInputDict[data].type() : ", self.trainInputDict["data"].type())
		# print("trainInputDict[label].shape : ", self.trainInputDict["label"].shape, " trainInputDict[label].type() : ", self.trainInputDict["label"].type())

	
	def work(self):
		labelsList = []
		labelMap = {}
		reverseLabelMap = {}
		self.trainInputDict["label"] = self.trainInputDict["label"].to(self.device)
		self.trainInputDict["data"] = self.trainInputDict["data"].to(self.device)

		oldData = self.trainInputDict["data"]
		oldLabel = self.trainInputDict["label"]
		# if self.parentId != 0:
		if self.isTrain:
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

			newData, newLabel = self.balanceData();
			self.trainInputDict["data"] = newData
			self.trainInputDict["label"] = newLabel

		self.cnnModel.to(self.device)
		self.mlpModel.to(self.device)
		if self.isTrain:
			self.trainCNN(labelMap, reverseLabelMap)
			print("CNN trained successfully...")
		

		

		if self.isTrain and not self.isLeaf:
			final_dict = {}
			ckpt = torch.load('ckpt/node_cnn_'+str(self.parentId)+'_'+str(self.nodeId)+'.pth')
			self.cnnModel.load_state_dict(ckpt['model_state_dict'])
			self.cnnModel.eval()

			# self.trainInputDict["data"].to(self.device)
			image_next, image_next_flat, _, _ = self.cnnModel(self.trainInputDict["data"].to(self.device))
			# image_next, image_next_flat, _ = self.cnnModel(self.trainInputDict["data"].to(self.device))
			image_next = image_next.detach()
			image_next_flat = image_next_flat.detach()
			img_flat_nmpy = image_next_flat.to("cpu")
			print("image_next_flat.shape : ", image_next_flat.shape)
			# cluster_ids, _ = kmeans_output(image_next_flat, self.device)
			img_flat_nmpy = img_flat_nmpy.numpy()
			# print("img_flat_nmpy.shape : " ,img_flat_nmpy.shape)
			# print("Starting K-Means...")
			kmeans = KMeans(n_clusters=2, n_jobs=-1).fit(img_flat_nmpy)
			# print("K-Means successfully completed...")
			# print("len(kmeans.labels_) : ", len(kmeans.labels_))
			#TODO: do 
			cluster_ids = kmeans.labels_

			leftCnt = {}
			rightCnt = {}


			for i in range(len(self.trainInputDict["data"])):
				label = self.trainInputDict["label"][i].item()
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

			for label, count in leftCnt.items():
				if label in rightCnt:
					if count >= rightCnt[label]:
						expected_dict[label] = 0
					else:
						expected_dict[label] = 1
				else:
					expected_dict[label] = 0

			for label, count in rightCnt.items():
				if not (label in expected_dict):
					expected_dict[label] = 1


			print("printing expected split from k means")
			print(expected_dict)                


			leftSortedListOfTuples = sorted(leftCnt.items(), reverse=True, key=lambda x: x[1])
			rightSortedListOfTuples = sorted(rightCnt.items(), reverse=True, key=lambda x: x[1])

			for ind, element in enumerate(leftSortedListOfTuples):
				if ind >= self.numClasses/2:
					final_dict[element[0]] = 1
				else:
					final_dict[element[0]] = 0  

			for ind, element in enumerate(rightSortedListOfTuples):
				if ind >= self.numClasses/2:
					final_dict[element[0]] = 0
				else:
					final_dict[element[0]] = 1             

			# for key, value in rightCnt.items():
			#     if not (key in final_dict):
			#         final_dict[key] = 1
			# final_dict=expected_dict
			print("Printing final_dict items...")
			print(final_dict)
			# for key, value in final_dict.items():
			#     print(key, " ", value)

			#TODO: separate for validation set too
			torch.save({
					'splittingDict':final_dict,
					}, 'ckpt/node_split_'+str(self.parentId)+'_'+str(self.nodeId)+'.pth')

			expectedMlpLabels = []
			for i in range(len(self.trainInputDict["data"])):
				label = self.trainInputDict["label"][i].item()
				expectedMlpLabels.append(final_dict[label])
			expectedMlpLabels = torch.tensor(expectedMlpLabels, device=self.device)
			print("expectedMlpLabels.shape : ",expectedMlpLabels.shape)
		
		
		if self.isTrain and not self.isLeaf:
			self.trainMLP(image_next_flat, expectedMlpLabels)
			print("MLP trained successfully...")

		final_dict = {}
		if not self.isLeaf:
			ckpt = torch.load('ckpt/node_split_'+str(self.parentId)+'_'+str(self.nodeId)+'.pth')
			final_dict = ckpt['splittingDict']
		#doing CNN on original tensors as those only goes to children
		self.trainInputDict["data"] = oldData
		self.trainInputDict["label"] = oldLabel
		ckpt = torch.load('ckpt/node_cnn_'+str(self.parentId)+'_'+str(self.nodeId)+'.pth')
		reverseLabelMap = ckpt['reverseLabelMap']
		labelMap = ckpt['labelMap']
		imgSize = self.trainInputDict["data"][0].shape[2]
		inChannels = self.trainInputDict["data"][0].shape[0]
		# print("nodeId: ", self.nodeId, ", imgSize : ", imgSize)
		outChannels = 16
		kernel = 5
		self.cnnModel = CNN(img_size=imgSize, in_channels=inChannels, out_channels=outChannels, num_class=len(reverseLabelMap), kernel=kernel)
		self.cnnModel.load_state_dict(ckpt['model_state_dict'])
		self.cnnModel.eval()
		self.cnnModel.to(self.device)

		# self.trainInputDict["data"].to(self.device)
		# image_next, image_next_flat, est_labels = self.cnnModel(self.trainInputDict["data"].to(self.device))
		image_next, image_next_flat, est_labels, _ = self.cnnModel(self.trainInputDict["data"].to(self.device))

		if not self.isTrain:
			_, predicted = est_labels.max(1)
			predicted = predicted.to(self.device)
			for i, val in enumerate(predicted):
				predicted[i] = reverseLabelMap[val.item()]

			correct = predicted.eq(self.trainInputDict["label"].to(self.device)).sum().item()
			total = len(est_labels)

			if not self.isLeaf:
				print('Root Node Acc: %.3f'% (100.*correct/total))
			
			if self.isLeaf:
				ckpt = torch.load('ckpt/testPred.pth')
				testPredDict = ckpt['testPredDict']
				testPredDict['actual'] = testPredDict['actual'].to(self.device)
				testPredDict['pred'] = testPredDict['pred'].to(self.device)
				# print(testPredDict['pred'].dtype, testPredDict['actual'].dtype, self.trainInputDict["label"].dtype)
				testPredDict['actual'] = torch.cat((testPredDict['actual'],self.trainInputDict["label"]),0)
				testPredDict['pred'] = torch.cat((testPredDict['pred'],predicted),0)
				torch.save({
					'testPredDict':testPredDict,
					}, 'ckpt/testPred.pth')


		if self.isLeaf:
			return

		ckpt = torch.load('ckpt/node_mlp_'+str(self.parentId)+'_'+str(self.nodeId)+'.pth')
		self.mlpModel.load_state_dict(ckpt['model_state_dict'])
		self.mlpModel.eval()
		self.mlpModel.to(self.device)

		est_labels = self.mlpModel(image_next_flat)
		# print(est_labels.shape)
		est_labels = est_labels.view(-1)
		mlpPrediction = est_labels.detach()
		mlpPrediction += 0.5
		mlpPrediction = mlpPrediction.long()

		trainLimages = []
		trainRimages = []
		trainLLabels = []
		trainRLabels = []
		lclasses = [0]*10
		rclasses = [0]*10
		for i, val in enumerate(mlpPrediction):
			# print("i : ", i)
			if val<=0.5:
				# trainLimages.append((image_next[i].detach()).tolist())
				lclasses[self.trainInputDict["label"][i].item()]+=1
				# trainLLabels.append(self.trainInputDict["label"][i].item())
			else:
				# trainRimages.append((image_next[i].detach()).tolist())
				rclasses[self.trainInputDict["label"][i].item()]+=1
				# trainRLabels.append(self.trainInputDict["label"][i].item())

		# lTrainDict = {"data":torch.tensor(trainLimages), "label":torch.tensor(trainLLabels)}
		# rTrainDict = {"data":torch.tensor(trainRimages), "label":torch.tensor(trainRLabels)}

		print(final_dict)
		totalLeftImages = 0.0
		totalRightImages = 0.0
		maxLeftClasses = 0.0
		maxRightClasses = 0.0
		testCorrectResults = 0.0
		for i, val in enumerate(lclasses):
			totalLeftImages += val
			if not self.isTrain and (i in labelMap) and final_dict[labelMap[i]] == 0:
				testCorrectResults += val
			maxLeftClasses = max(maxLeftClasses, val)

		for i, val in enumerate(rclasses):
			totalRightImages += val
			if not self.isTrain and (i in labelMap) and final_dict[labelMap[i]] == 1:
				testCorrectResults += val
			maxRightClasses = max(maxRightClasses, val)

		if not self.isTrain:
			total = float(len(self.trainInputDict["label"]))
			print('Split Acc: %.3f'% (100.*testCorrectResults/total))

		leftClassesToBeRemoved = []
		rightClassesToBeRemoved = []

		threshold = 10.0
		for i, val in enumerate(lclasses):
			if float(100*val)/maxLeftClasses < threshold:
				leftClassesToBeRemoved.append(i)

		for i, val in enumerate(rclasses):
			if float(100*val)/maxRightClasses < threshold:
				rightClassesToBeRemoved.append(i)

		lclasses = [0]*10
		rclasses = [0]*10
		for i, val in enumerate(mlpPrediction):
			# print("i : ", i)
			if val<=0.5:
				if self.isTrain:
					if not (self.trainInputDict["label"][i].item() in leftClassesToBeRemoved):
						trainLimages.append((image_next[i].detach()).tolist())
						lclasses[self.trainInputDict["label"][i].item()]+=1
						trainLLabels.append(reverseLabelMap[self.trainInputDict["label"][i].item()])
				else:
					trainLimages.append((image_next[i].detach()).tolist())
					lclasses[self.trainInputDict["label"][i].item()]+=1
					trainLLabels.append(self.trainInputDict["label"][i].item())
			else:
				if self.isTrain:
					if not (self.trainInputDict["label"][i].item() in rightClassesToBeRemoved):
						trainRimages.append((image_next[i].detach()).tolist())
						rclasses[self.trainInputDict["label"][i].item()]+=1
						trainRLabels.append(reverseLabelMap[self.trainInputDict["label"][i].item()])
				else:
					trainRimages.append((image_next[i].detach()).tolist())
					rclasses[self.trainInputDict["label"][i].item()]+=1
					trainRLabels.append(self.trainInputDict["label"][i].item())

		lTrainDict = {"data":torch.tensor(trainLimages), "label":torch.tensor(trainLLabels)}
		rTrainDict = {"data":torch.tensor(trainRimages), "label":torch.tensor(trainRLabels)}

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

		print("---")
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

		print("impurityDrop: ", impurityDrop)
		print("giniGain: ", self.giniValue - impurityDrop)
		print("lclasses: ", lclasses)
		print("rclasses: ", rclasses)
		print("noOfLeftClasses: ", noOfLeftClasses)
		print("noOfRightClasses: ", noOfRightClasses)
		print("lTrainDict[data].shape: ", lTrainDict["data"].shape, "  lTrainDict[label].shape: ", lTrainDict["label"].shape)
		print("rTrainDict[data].shape: ", rTrainDict["data"].shape, "  rTrainDict[label].shape: ", rTrainDict["label"].shape)

		lValDict = {}
		rValDict = {}
		print("RETURNING FROM WORK...")
		#TODO: populate validation dictionaries too
		# return lTrainDict, lValDict, rTrainDict, rValDict, giniLeftRatio, giniRightRatio
		if self.isTrain and not self.isLeaf:
			return lTrainDict, lValDict, rTrainDict, rValDict, giniLeftRatio, giniRightRatio, noOfLeftClasses, noOfRightClasses
		elif not self.isTrain and not self.isLeaf:
			return lTrainDict, rTrainDict, giniLeftRatio, giniRightRatio, noOfLeftClasses, noOfRightClasses
		elif self.isTrain and self.isLeaf:
				return ""
		else:
			return ""

