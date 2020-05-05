class Node():

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

        newData, newLabel = self.balanceData();
        self.trainInputDict["data"] = newData
        self.trainInputDict["label"] = newLabel
        return labelMap, reverseLabelMap

    def loadMLPModel(self):
        ckpt = torch.load('ckpt/node_mlp_'+str(self.parentId)+'_'+str(self.nodeId)+'.pth')
        self.mlpModel.load_state_dict(ckpt['model_state_dict'])
        self.mlpModel.eval()
        self.mlpModel.to(self.device)

    def loadCNNModel(self):
        ckpt = torch.load('ckpt/node_cnn_'+str(self.parentId)+'_'+str(self.nodeId)+'.pth')
        self.cnnModel.load_state_dict(ckpt['model_state_dict'])
        self.cnnModel.eval()
        self.cnnModel.to(self.device)
        return ckpt['reverseLabelMap']

    def separateLabels(self, cluster_ids):
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
        return leftSortedListOfTuples, rightSortedListOfTuples

    def makeFinalDict(self, leftSortedListOfTuples, rightSortedListOfTuples):
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

        print("Printing final_dict items...")
        print(final_dict)

        #TODO: separate for validation set too
        torch.save({
                'splittingDict':final_dict,
                }, 'ckpt/node_split_'+str(self.parentId)+'_'+str(self.nodeId)+'.pth')
        return final_dict

    def getSavedFinalSplit(self):
        ckpt = torch.load('ckpt/node_split_'+str(self.parentId)+'_'+str(self.nodeId)+'.pth')
		return ckpt['splittingDict']

    def setFinalPredictions(self):
        ckpt = torch.load('ckpt/testPred.pth')
        testPredDict = ckpt['testPredDict']
        testPredDict['actual'] = testPredDict['actual'].to(self.device)
        testPredDict['pred'] = testPredDict['pred'].to(self.device)
        testPredDict['actual'] = torch.cat((testPredDict['actual'],self.trainInputDict["label"].to(self.device)),0)
        testPredDict['pred'] = torch.cat((testPredDict['pred'],predicted),0)
        torch.save({
            'testPredDict':testPredDict,
            }, 'ckpt/testPred.pth')

    def checkTestPreds(self, reverseLabelMap, est_labels):
        _, predicted = est_labels.max(1)
        predicted = predicted.to(self.device)
        for i, val in enumerate(predicted):
            predicted[i] = reverseLabelMap[val.item()]

        correct = predicted.eq(self.trainInputDict["label"].to(self.device)).sum().item()
        total = len(est_labels)

        if not self.isLeaf:
            print('Root Node Acc: %.3f'% (100.*correct/total))
        
        if self.isLeaf:
            self.setFinalPredictions()

    def doLabelCounting(self, mlpPrediction):
        lclasses = [0]*10
		rclasses = [0]*10

		for i, val in enumerate(mlpPrediction):
			if val<=0.5:
				lclasses[self.trainInputDict["label"][i].item()]+=1
			else:
				rclasses[self.trainInputDict["label"][i].item()]+=1
        retutn lclasses, rclasses
    
    def countBalanceAndThreshold(self, mlpPrediction):
        final_dict = self.getSavedFinalSplit()
        lclasses, rclasses = self.doLabelCounting(mlpPrediction)
		totalLeftImages = 0.0
		totalRightImages = 0.0
		maxLeftClasses = 0.0
		maxRightClasses = 0.0
		testCorrectResults = 0.0
		for i, val in enumerate(lclasses):
			totalLeftImages += val
			if not self.isTrain and final_dict[i] == 0:
				testCorrectResults += val
			maxLeftClasses = max(maxLeftClasses, val)

		for i, val in enumerate(rclasses):
			totalRightImages += val
			if not self.isTrain and final_dict[i] == 1:
				testCorrectResults += val
			maxRightClasses = max(maxRightClasses, val)

		if not self.isTrain:
			total = float(len(self.trainInputDict["label"]))
			print('Split Acc: %.3f'% (100.*testCorrectResults/total))

		leftClassesToBeRemoved = []
		rightClassesToBeRemoved = []

		threshold = 1.0
		for i, val in enumerate(lclasses):
			if float(100*val)/maxLeftClasses < threshold:
				leftClassesToBeRemoved.append(i)

		for i, val in enumerate(rclasses):
			if float(100*val)/maxRightClasses < threshold:
				rightClassesToBeRemoved.append(i)
        
        return totalLeftImages, totalRightImages, maxLeftClasses, maxRightClasses, testCorrectResults, leftClassesToBeRemoved, rightClassesToBeRemoved

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
		


    def classifyLabels(self, mlpPrediction):
        totalLeftImages, totalRightImages, maxLeftClasses, maxRightClasses, testCorrectResults, leftClassesToBeRemoved, rightClassesToBeRemoved = self.countBalanceAndThreshold(mlpPrediction, final_dict)
        trainLimages = []
		trainRimages = []
		trainLLabels = []
		trainRLabels = []
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
					trainLLabels.append(reverseLabelMap[self.trainInputDict["label"][i].item()])
			else:
				if self.isTrain:
					if not (self.trainInputDict["label"][i].item() in rightClassesToBeRemoved):
						trainRimages.append((image_next[i].detach()).tolist())
						rclasses[self.trainInputDict["label"][i].item()]+=1
						trainRLabels.append(reverseLabelMap[self.trainInputDict["label"][i].item()])
				else:
					trainRimages.append((image_next[i].detach()).tolist())
					rclasses[self.trainInputDict["label"][i].item()]+=1
					trainRLabels.append(reverseLabelMap[self.trainInputDict["label"][i].item()])

		lTrainDict = {"data":torch.tensor(trainLimages), "label":torch.tensor(trainLLabels)}
		rTrainDict = {"data":torch.tensor(trainRimages), "label":torch.tensor(trainRLabels)}
        
		self.getPredictionAnalysis(totalLeftImages, totalRightImages, lclasses, rclasses)
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


    def work_test(self):
		reverseLabelMap = self.loadCNNModel()
		image_next, image_next_flat, est_labels = self.cnnModel(self.trainInputDict["data"].to(self.device))

        if not self.isTrain:
            self.checkTestPreds(reverseLabelMap, est_labels)
            
        if self.isLeaf:
			return

		self.loadMLPModel()

		est_labels = self.mlpModel(image_next_flat)
		est_labels = est_labels.view(-1)
		mlpPrediction = est_labels.detach()
		mlpPrediction += 0.5
		mlpPrediction = mlpPrediction.long()
        return self.classifyLabels(mlpPrediction)


    def workTrain(self):
		oldData = self.trainInputDict["data"]
		oldLabel = self.trainInputDict["label"]
        labelMap, reverseLabelMap = self.make_labels_list()			

		self.cnnModel.to(self.device)
        self.trainCNN(labelMap, reverseLabelMap)
        print("CNN trained successfully...")		

		if not self.isLeaf:
    		self.loadCNNModel()
			final_dict = {}
			image_next, image_next_flat, _ = self.cnnModel(self.trainInputDict["data"].to(self.device))
			image_next = image_next.detach()
			image_next_flat = image_next_flat.detach()
			img_flat_nmpy = image_next_flat.to("cpu")
			print("image_next_flat.shape : ", image_next_flat.shape)
			img_flat_nmpy = img_flat_nmpy.numpy()
			kmeans = KMeans(n_clusters=2, n_jobs=-1).fit(img_flat_nmpy)
			cluster_ids = kmeans.labels_
            leftSortedListOfTuples, rightSortedListOfTuples = self.separateLabels(cluster_ids)
			final_dict = self.makeFinalDict(leftSortedListOfTuples, rightSortedListOfTuples)

			expectedMlpLabels = []
			for i in range(len(self.trainInputDict["data"])):
				label = self.trainInputDict["label"][i].item()
				expectedMlpLabels.append(final_dict[label])
			expectedMlpLabels = torch.tensor(expectedMlpLabels, device=self.device)
			print("expectedMlpLabels.shape : ",expectedMlpLabels.shape)
		
		if not self.isLeaf:
            self.mlpModel.to(self.device)
			self.trainMLP(image_next_flat, expectedMlpLabels)
			print("MLP trained successfully...")
        self.trainInputDict["data"] = oldData
		self.trainInputDict["label"] = oldLabel

        return self.work_test()
