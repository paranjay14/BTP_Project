import torch
import torch.nn as nn
from cnn import CNN
'''     # comment this line if want to use big_MLP architecture instead of small_MLP 
from mlp import MLP
'''
from mlp_small import MLP
# '''
import time
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import smote_variants as sv
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from getOptions import options
from torchsummary import summary

# this isn't used currently : kmeans function
def kmeans_output(all_images_flat, device, num_clusters=2):
    cluster_ids_x, cluster_centers = kmeans(X=all_images_flat, num_clusters=num_clusters, distance='euclidean', device=device)
    return cluster_ids_x, cluster_centers

# class of Node
class myNode:
    # constructor of node, having parent id, own id, bool to tell to train or not, level of node, append itself in parent node
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

    # set inputs of node as train/test dictionaries, validation input(for test it will be 0 tensor), num of classes of data in this node,
    # ginivalue of parent, if node is leaf or noe, left and right child node ids respectively
    def setInput(self, trainInputDict, valInputDict, numClasses, giniValue, isLeaf, leafClass, lchildId, rchildId):
        self.trainInputDict = trainInputDict
        self.valInputDict = valInputDict
        imgSize = trainInputDict["data"][0].shape[2]
        inChannels = trainInputDict["data"][0].shape[0]
        if options.verbose > 0:
            print("Setting Input for nodeId: "+str(self.nodeId)+"...")
        if options.verbose > 1:
            print("imgTensorShape : ", trainInputDict["data"].shape)
        outChannels = options.cnnOut
        outChannels = options.cnnOut + 8*(self.level)
        kernel = 5
        # CNN architecture is selected accoding to the input channels
        self.cnnModel = CNN(img_size=imgSize, in_channels=inChannels, out_channels=outChannels, num_class=numClasses, kernel=kernel, use_bn=False)
        numFeatures = self.cnnModel.features
        # mlp model is similarly constructed
        self.mlpModel = MLP(numFeatures, use_bn=False)
        ## self.dtModel = DecisionTreeClassifier(random_state=0, criterion='gini', splitter='best', max_depth=1, max_features=None, min_samples_split=2, min_samples_leaf=1)
        self.dtModel = DecisionTreeRegressor(random_state=0, criterion='mse', splitter='best', max_depth=2, max_features=None, min_samples_split=2, min_samples_leaf=1)
        ## self.rfModel = RandomForestClassifier(random_state=0, criterion='gini', n_estimators=100, max_depth=1, max_features='auto', min_samples_split=2, min_samples_leaf=1)
        self.rfModel = RandomForestRegressor(random_state=0, criterion='mse', n_estimators=5, max_depth=2, max_features='auto', min_samples_split=2, min_samples_leaf=1)
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
        if options.verbose > 1:
            print("nodeId:", self.nodeId, ",  parentId:", self.parentId, ",  level:", self.level, ",  lchildId:", self.lchildId, ",  rchildId:", self.rchildId, ",  isLeaf:", self.isLeaf, ",  leafClass:", self.leafClass, ",  numClasses:", self.numClasses, ",  numData:", self.numData)

    # function to train CNN
    def trainCNN(self, labelMap, reverseLabelMap):        
        loss_fn = nn.CrossEntropyLoss()     # used cross entropy loss function
        ## loss_fn_mse = nn.MSELoss()        
        optimizer = torch.optim.Adam(self.cnnModel.parameters(),lr=options.cnnLR)   # adam optimiser used
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, options.cnnSchEpochs, options.cnnSchFactor)      # used step LR 
        ## scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.2, mode='max',verbose=True)
        self.cnnModel.to(self.device)
        trainLabels = self.trainInputDict["label"]
        trainInputs = self.trainInputDict["data"]

        # set number of batches and get batch size accordingly
        numBatches = options.cnnBatches
        batchSize = int((len(trainInputs))/(numBatches-1))
        if(batchSize<3):
            batchSize=int(len(trainInputs))
            numBatches=1
        # set number of epochs
        numEpochs = options.cnnEpochs - 10*(self.level)
        # numEpochs = max(30, options.cnnEpochs - 10*(self.level))
        # numEpochs = options.cnnEpochs

        st_btch = 0
        batch_sep = []      # batch_sep stores the pairs of (st_btch,end_btch), and is randomly shuffled to reduce chances of overfitting
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
            self.cnnModel.train()   # sets CNN to training mode
            total = 0
            correct = 0
            train_loss = 0
            random.shuffle(batch_sep)
            for batch in range(numBatches):
                st_btch, end_btch = batch_sep[batch]
                optimizer.zero_grad()   # reset
                _, _, est_labels, feat_same = self.cnnModel(trainInputs[st_btch:end_btch].to(self.device))  # CNN model is trained
                batch_loss_label = loss_fn(est_labels, trainLabels[st_btch:end_btch].to(self.device))
                ## batch_loss_featr = loss_fn_mse(feat_same, trainInputs[st_btch:end_btch])
                ## batch_loss = batch_loss_featr + batch_loss_label
                batch_loss =  batch_loss_label
                batch_loss.backward()
                optimizer.step()
                ## if options.verbose > 1:
                ##     print(batch_loss_featr.item(), batch_loss_label.item())
                ## if batch == 0:
                ##     train_loss_tensor = batch_loss
                ## else:
                ##     train_loss_tensor += batch_loss
                train_loss += batch_loss.item()
                _, predicted = est_labels.max(1)
                total += end_btch - st_btch
                correct += predicted.eq(trainLabels[st_btch:end_btch].to(self.device)).sum().item()
            scheduler.step()
            # scheduler.step(train_loss_tensor)
            # train_loss = train_loss_tensor.item()

            ''' ## If using validation, just comment this line by prepending with a #, else remove any hashes present in the beginning
            if (epoch%options.cnnSchEpochs == options.cnnSchEpochs-1) or (epoch%options.cnnSchEpochs == 0):
                self.cnnModel.eval()
                _, _, est_labels, _ = self.cnnModel(self.valInputDict["data"].to(self.device))
                val_loss = loss_fn(est_labels, self.valInputDict["label"].to(self.device))
                _, predicted = est_labels.max(1)
                valTotalSize = float(len(self.valInputDict["data"]))
                valCorrect = predicted.eq(self.valInputDict["label"].to(self.device)).sum().item()
                scheduler.step(valCorrect)
                if options.verbose > 1:
                    print(epoch, 'Train Loss: %.3f | Train Acc: %.3f | Val Loss: %.3f | Val Accuracy: %.3f'% (train_loss, 100.*correct/total, val_loss.item(), 100.*float(valCorrect)/valTotalSize))
            # '''
            train_loss /= numBatches
            if total != 0:
                if options.verbose > 1:
                    print(epoch, 'Train Loss: %.3f | Train Acc: %.3f '% (train_loss, 100.*correct/total))

        epoch = numEpochs
        torch.save({
                'epoch':epoch,
                'model_state_dict':self.cnnModel.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'train_loss':train_loss,
                'labelMap':labelMap,
                'reverseLabelMap':reverseLabelMap,  
                }, options.ckptDir+'/node_cnn_'+str(self.nodeId)+'.pth')


    # function to train MLP
    def trainMLP(self, trainInputs, trainTargets, weightVector):
        # used BCE loss function and adam optimiser with step learning rate
        loss_fn = nn.BCELoss(reduction='none')
        optimizer = torch.optim.Adam(self.mlpModel.parameters(),lr=options.mlpLR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, options.mlpSchEpochs, options.mlpSchFactor)

        weightVector = weightVector.to(self.device)
        self.mlpModel.to(self.device)

        # set number of batches and get batch size accordingly
        numBatches = options.mlpBatches
        batchSize = int((len(trainInputs))/(numBatches-1))
        if(batchSize<3):
            batchSize=int(len(trainInputs))
            numBatches=1

        st_btch = 0
        batch_sep = []

        # construct baches start and end number and store them in batch_sep
        for i in range(numBatches):
            end_btch = min(st_btch + batchSize, len(trainInputs))
            if end_btch == st_btch:
                numBatches-=1
                break
            else:
                batch_sep.append([st_btch, end_btch])
                st_btch = end_btch

        # set number of epochs
        numEpochs = options.mlpEpochs
        ## numEpochs = max(30, options.mlpEpochs-10*(self.level))
        self.mlpModel.train()
        train_loss=0
        # for each epoch, train MLP in batches
        for epoch in range(numEpochs):
            train_loss = 0
            correct = 0
            total = 0
            random.shuffle(batch_sep)

            for batch in range(numBatches):    
                st_btch, end_btch = batch_sep[batch]
                optimizer.zero_grad()
                est_labels = self.mlpModel(trainInputs[st_btch:end_btch].to(self.device))
                est_labels = est_labels.view(-1)
                ## batch_loss = loss_fn(est_labels, trainTargets[st_btch:end_btch]) #if crossentropy
                # weighted loss
                batch_loss = weightVector[st_btch:end_btch] * loss_fn(est_labels, trainTargets[st_btch:end_btch].float().to(self.device)) #if bce
                batch_loss = batch_loss.mean()
                batch_loss.backward()
                optimizer.step()

                train_loss += batch_loss.item()
                ## _, predicted = est_labels.max(1) # if cross entropy
                predicted = est_labels.detach() #if bce
                predicted += 0.5
                predicted = predicted.long()
                ## total += trainTargets.size(0)
                total += end_btch - st_btch
                correct += predicted.eq(trainTargets[st_btch:end_btch].to(self.device)).sum().item()
            scheduler.step()
            
            if total != 0:
                if options.verbose > 1:
                    print(epoch, 'Loss: %.3f | Acc: %.3f'% (train_loss, 100.*correct/total))

        epoch = numEpochs
        torch.save({
                'epoch':epoch,
                'model_state_dict':self.mlpModel.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'train_loss':train_loss,
                }, options.ckptDir+'/node_mlp_'+str(self.nodeId)+'.pth')


    # function to train Decision Tree (DT)
    def trainDT(self, trainInputs, trainTargets, weightVector):
        self.dtModel = self.dtModel.fit(trainInputs.numpy(), trainTargets.numpy(), weightVector.numpy())
        if options.verbose > 1:
            print(' n_leaves:', self.dtModel.get_n_leaves(), ' depth:', self.dtModel.get_depth())

        torch.save({
                'n_leaves': self.dtModel.get_n_leaves(),
                'depth': self.dtModel.get_depth(),
                'params': self.dtModel.get_params(),
                'model_state_dict': self.dtModel,
                }, options.ckptDir+'/node_dt_'+str(self.nodeId)+'.pth')


    # function to train Random Forest (RF)
    def trainRF(self, trainInputs, trainTargets, weightVector):
        self.rfModel = self.rfModel.fit(trainInputs.numpy(), trainTargets.numpy(), weightVector.numpy())
        if options.verbose > 1:
            print('params:', self.rfModel.get_params())

        torch.save({
                'params': self.rfModel.get_params(),
                'model_state_dict': self.rfModel,
                }, options.ckptDir+'/node_rf_'+str(self.nodeId)+'.pth')



    # this is required if input data doesn't have same data points in each class while training the node
    # (but this is not the case in our current implementation when using {options.caseNum as 1 or 2})
    def balanceData(self):
        shape=self.trainInputDict["data"].shape
        if options.verbose > 1:
            print("trainInputDict[data].shape : ", shape)
        copy = self.trainInputDict["data"]
        copy = copy.reshape(shape[0], -1)
        if options.verbose > 1:
            print("copy.shape : ", copy.shape)
        npDict = copy.numpy()
        copyLabel = self.trainInputDict["label"]
        if options.verbose > 1:
            print("copyLabel.shape : ", copyLabel.shape)
        ## copyLabel = copyLabel.view(-1)
        npLabel = copyLabel.numpy()
        ## X_resampled, y_resampled = kmeans_smote.fit_sample(npDict, npLabel)

        ## if options.verbose > 1:
            ## print(sv.get_all_oversamplers_multiclass())

        oversampler= sv.MulticlassOversampling(sv.SMOTE(n_jobs=6))

        X_resampled, y_resampled = oversampler.sample(npDict, npLabel)
        if options.verbose > 1:
            [print('Class {} has {} instances after oversampling'.format(label, count)) for label, count in zip(*np.unique(y_resampled, return_counts=True))]

        newData = torch.from_numpy(X_resampled.reshape(len(X_resampled), shape[1], shape[2], shape[3]))
        newLabel = torch.from_numpy(y_resampled)
        newData = newData.float()
        return newData, newLabel

    # makes labels map and reverse label map, which is required because CNN output labels should be 0 indexed
    # but the labels of input data can be 4,2,5,6, which will be mapped to 0,1,2,3 for CNN training and stored in labelMap,
    # reverse of above will be stores in reverseLabelMap, which will be 0,1,2,3 -> 4,2,5,6
    # hence accordingly input labels are changed to required labels
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

        ''' # if we have to balance the data, then comment the current line by prepending a hash(#)
        newData, newLabel = self.balanceData();
        self.trainInputDict["data"] = newData
        self.trainInputDict["label"] = newLabel
        # '''
        return labelMap, reverseLabelMap

    # loads CNN model from stored dictionary
    def loadCNNModel(self):
        ckpt = torch.load(options.ckptDir+'/node_cnn_'+str(self.nodeId)+'.pth')
        self.cnnModel.load_state_dict(ckpt['model_state_dict'])
        self.cnnModel.eval()
        self.cnnModel.to(self.device)
        return ckpt['reverseLabelMap'], ckpt['labelMap']

    # loads MLP model from stored dictionary
    def loadMLPModel(self):
        ckpt = torch.load(options.ckptDir+'/node_mlp_'+str(self.nodeId)+'.pth')
        self.mlpModel.load_state_dict(ckpt['model_state_dict'])
        self.mlpModel.eval()
        self.mlpModel.to(self.device)

    # loads DT model from stored dictionary
    def loadDTModel(self):
        ckpt = torch.load(options.ckptDir+'/node_dt_'+str(self.nodeId)+'.pth')
        self.dtModel = ckpt['model_state_dict']

    # loads RF model from stored dictionary
    def loadRFModel(self):
        ckpt = torch.load(options.ckptDir+'/node_rf_'+str(self.nodeId)+'.pth')
        self.rfModel = ckpt['model_state_dict']

    # This constructs the dictionaries where data points of each class will go on 2-means clustering output
    def separateLabels(self, cluster_ids):
        # first we count for each class how many data points belong to left cluster and right cluster respectively and
        # we store them in leftCnt and rightCnt respectively
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

        # traversing leftCnt dict. and seeing its intersections with rightCnt dict.
        for label, count in leftCnt.items():
            lRatio=1.0
            if label in rightCnt:
                lRatio = float(count)/float(count+rightCnt[label])
                if lRatio>maxLeftRatio:
                    maxLeftRatio = lRatio
                    lIndx = label
                ## rRatio = float(rightCnt[label])/float(count+rightCnt[label])
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

        # traversing rightCnt dict. and seeing its difference with leftCnt dict.
        for label, count in rightCnt.items():
            if not (label in expected_dict):
                rRatio=1.0
                if rRatio>maxRightRatio:
                    maxRightRatio = rRatio
                    rIndx = label
                expected_dict[label] = 1
                rCnt+=1

        # if all samples went to the right side, forcefully send the class having maximum lRatio (maxLeftRatio) to the left side 
        if lCnt==0:
            if options.verbose > 0:
                print("L_CNT=0 detected")
            expected_dict[lIndx] = 0
        # if all samples went to the left side, forcefully send the class having maximum rRatio (maxRightRatio) to the right side 
        if rCnt==0:
            if options.verbose > 0:
                print("R_CNT=0 detected")
            expected_dict[rIndx] = 1

        if options.verbose > 1:
            print("printing expected split from k means")
            print(expected_dict)  
        # created and return a sorted list of tuples to be used later while making final_dict              
        leftSortedListOfTuples = sorted(leftCnt.items(), reverse=True, key=lambda x: x[1])
        rightSortedListOfTuples = sorted(rightCnt.items(), reverse=True, key=lambda x: x[1])
        return leftSortedListOfTuples, rightSortedListOfTuples, expected_dict

    # creates the final dictionary to be used 
    def makeFinalDict(self, leftSortedListOfTuples, rightSortedListOfTuples, expected_dict):
        final_dict = {}

        # use the expected dict. as it is 
        if options.caseNum == 1:
            final_dict = expected_dict

        # create a final dict. having half classes(along with their data points) in both children sides when the parent node contains even no. of diff. classes  
        # while, when the parent node contains an odd no. of classes, then the right child gets one extra class data points' as compared to the left one
        elif options.caseNum == 2:
            fullDict = {}
            for ind, element in enumerate(leftSortedListOfTuples):
                fullDict[element[0]] = element[1]
                if ind >= self.numClasses/2:
                    final_dict[element[0]] = 1
                else:
                    final_dict[element[0]] = 0  

            for ind, element in enumerate(rightSortedListOfTuples):
                if element[0] not in fullDict:
                    fullDict[element[0]] = -1*element[1]

                if ind >= self.numClasses/2:
                    final_dict[element[0]] = 0
                else:
                    final_dict[element[0]] = 1        

            fullSortedListOfTuples = sorted(fullDict.items(), reverse=True, key=lambda x: (x[1],x[0]))
            # This below code should be used only if  we want both the children nodes to contain exactly same no. of classes 
            # even in case of odd no. of classes present in the parent node. Here, we distribute half-half elements of a single particular class to both children
            ## if (self.numClasses%2 == 1):
            ##  final_dict[fullSortedListOfTuples[int(self.numClasses/2)][0]] = -1

        if options.verbose > 1:
            print("Printing final_dict items...")
            print(final_dict)

        # save the final dict. in a file for using it later
        torch.save({
                'splittingDict':final_dict,
                }, options.ckptDir+'/node_split_'+str(self.nodeId)+'.pth')
        return final_dict

    # returns the saved final dict.
    def getSavedFinalSplit(self):
        ckpt = torch.load(options.ckptDir+'/node_split_'+str(self.nodeId)+'.pth')
        return ckpt['splittingDict']

    # saves the pred and act labels in a file along with thier indices
    def setFinalPredictions(self, predicted):
        ckpt = torch.load(options.ckptDir+'/testPred.pth')
        testPredDict = ckpt['testPredDict']
        testPredDict['actual'] = testPredDict['actual'].to(self.device)
        testPredDict['pred'] = testPredDict['pred'].to(self.device)
        testPredDict['index'] = testPredDict['index'].to(self.device)
        testPredDict['actual'] = torch.cat((testPredDict['actual'],self.trainInputDict["label"].to(self.device)),0)
        testPredDict['pred'] = torch.cat((testPredDict['pred'],predicted),0)
        testPredDict['index'] = torch.cat((testPredDict['index'],self.trainInputDict["index"].to(self.device)),0)

        torch.save({
            'testPredDict':testPredDict,
            }, options.ckptDir+'/testPred.pth')

    # this function compares the output we got with the actual labels
    def checkTestPreds(self, reverseLabelMap, est_labels, nodeProb, oneHotTensors):
        _, predicted = est_labels.max(1)    # gets the predicted label corresponding to the max probability value (per sample)
        predicted = predicted.to(self.device)
        # updates predicted labels to now store the corresponding labels according to reverseLabelMap
        for i, val in enumerate(predicted):
            predicted[i] = reverseLabelMap[val.item()]

        # calculates the no. of correct predicted labels
        correct = predicted.eq(self.trainInputDict["label"].to(self.device)).sum().item()
        total = len(est_labels)
        
        # nodeAcc (format) :: (correct, total, 100.*correct/total)
        self.nodeAcc[2] = 0.0
        if total != 0:
            self.nodeAcc[2] = round(100.*correct/total,3)
        self.nodeAcc[1] = total
        self.nodeAcc[0] = correct
        if options.verbose > 1:
            print('Node %d Acc: %.3f'% ( self.nodeId, self.nodeAcc[2]))
        
        # updates the nodeAcc in the file, to be used later while printing tree
        NodeDict = torch.load(options.ckptDir+'/node_'+str(self.nodeId)+'.pth')['nodeDict']
        NodeDict['nodeAcc'] = self.nodeAcc[:]
        torch.save({
            'nodeDict':NodeDict,
            }, options.ckptDir+'/node_'+str(self.nodeId)+'.pth')

        # when not using probabilistic method
        if not options.probabilistic:
            # Level Dict. stores the level-wise accuracies and for calculating this, it uses all the leaf accuracies that come across by this time too
            LevelDict = torch.load(options.ckptDir+'/level.pth')['levelDict']
            LevelDict['levelAcc'][self.level][0] += correct
            LevelDict['levelAcc'][self.level][1] += total
            if(self.isLeaf):
                LevelDict['leafAcc'][0] += correct
                LevelDict['leafAcc'][1] += total
            torch.save({
                'levelDict':LevelDict,
                }, options.ckptDir+'/level.pth')

        # if a leaf node
        if self.isLeaf:
            if self.level != 0:
                # calculate geometric mean of nodeProb seen so far from root to this leaf node
                nodeProb = nodeProb.pow(1/self.level)

            if options.probabilistic:
                # if probabilistic method is used, update the oneHotTensors corresponding to the predicted labels with their respective nodeProb 
                oneHotTensors[torch.arange(len(oneHotTensors)), predicted.long()] += nodeProb
            else:
                self.setFinalPredictions(predicted)

    # returns the indices of class labels going to left and right side
    def doLabelCounting(self, modelPrediction):
        lclasses = [0]*10 # 0-indexed according to self.trainInputDict["label"]
        rclasses = [0]*10

        for i, val in enumerate(modelPrediction):
            if val<=0.5:
                lclasses[self.trainInputDict["label"][i].item()]+=1
            else:
                rclasses[self.trainInputDict["label"][i].item()]+=1
        return lclasses, rclasses

    # used for calculating split accuracy and could also be used to remove redundant classes from a node
    def countBalanceAndThreshold(self, modelPrediction, labelMap):        
        final_dict = self.getSavedFinalSplit()      # loading saved final_dict from file
        lclasses, rclasses = self.doLabelCounting(modelPrediction)
        # totalLeftImages = 0.0
        # totalRightImages = 0.0
        maxLeftClasses = 0.0
        maxRightClasses = 0.0
        testCorrectResults = 0.0    # testCorrectResults stores the no. of test samples going to the correct side as was declared while training 
        totalL = 0
        totalR = 0

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
            # total = float(len(self.trainInputDict["label"]))      # all test samples at the current node
            total = float(totalL+totalR)    # all those samples at the current node whose classes participated during training at this numbered/indexed train_node only
            splitAcc = 0
            if total != 0:
                splitAcc = float(100.*testCorrectResults/total)
            if options.verbose > 1:
                print('Split Acc: %.3f'% (splitAcc))
            NodeDict = torch.load(options.ckptDir+'/node_'+str(self.nodeId)+'.pth')['nodeDict']
            NodeDict['splitAcc'] = splitAcc
            # saving the split acc to be used later while printing tree 
            torch.save({
                'nodeDict':NodeDict,
                }, options.ckptDir+'/node_'+str(self.nodeId)+'.pth')

        leftClassesToBeRemoved = []
        rightClassesToBeRemoved = []

        '''  comment this line, in order to use (NOT USED CURRENTLY) this code which removes those classes from a child node whose percantage in a node is less than the threshold value
        threshold = 15.0
        for i, val in enumerate(lclasses):
            if float(100*val)/maxLeftClasses < threshold:
                leftClassesToBeRemoved.append(i)

        for i, val in enumerate(rclasses):
            if float(100*val)/maxRightClasses < threshold:
                rightClassesToBeRemoved.append(i)
        #'''

        return maxLeftClasses, maxRightClasses, testCorrectResults, leftClassesToBeRemoved, rightClassesToBeRemoved

    # calculates the Gini_Gain for the current node (in relation to its children nodes) as well as gini values for both its children nodes
    def getPredictionAnalysis(self, totalLeftImages, totalRightImages, lclasses, rclasses):
        giniLeftRatio = 0.0
        giniRightRatio = 0.0
        lcheck = 0.0
        rcheck = 0.0
        if options.verbose > 1:
            print("# of Left images: ", totalLeftImages)
            print("# of Right images: ", totalRightImages)
        noOfLeftClasses = 0
        noOfRightClasses = 0
        for i in lclasses:
            if i != 0:
                noOfLeftClasses += 1
            pi=0
            if totalLeftImages != 0:
                pi = float(i)/totalLeftImages
            lcheck += pi
            giniLeftRatio += pi*(1-pi)

        for i in rclasses:
            if i != 0:
                noOfRightClasses += 1
            pi=0
            if totalRightImages != 0:
                pi = float(i)/totalRightImages
            rcheck += pi
            giniRightRatio += pi*(1-pi)

        if options.verbose > 1:
            print("giniRightRatio: ", giniRightRatio)
            print("giniLeftRatio: ", giniLeftRatio)

        leftChildrenRatio = 0
        if totalRightImages != 0:
            leftChildrenRatio = totalLeftImages/totalRightImages

        impurity = leftChildrenRatio*float(giniLeftRatio) + (1-leftChildrenRatio)*float(giniRightRatio)
        giniGain = self.giniValue - impurity

        if options.verbose > 1:
            print("impurity: ", impurity)
            print("giniGain: ", giniGain)
            print("lclasses: ", lclasses)
            print("rclasses: ", rclasses)
            print("noOfLeftClasses: ", noOfLeftClasses)
            print("noOfRightClasses: ", noOfRightClasses)
        return giniLeftRatio, giniRightRatio, noOfLeftClasses, noOfRightClasses, giniGain
        

    # separates the data to be send to left and right children nodes of the current node
    def classifyLabels(self, modelPrediction, reverseLabelMap, labelMap, lChildProb, rChildProb, oneHotTensors):
        totalLeftImages = 0.0
        totalRightImages = 0.0
        leftChildIndexList = []     # leftChildIndexList/rightChildIndexList store the indices of the data samples to the left/right side resp.
        rightChildIndexList = []
        leftValIndexList = []       # leftValIndexList/rightValIndexList store the indices of the valid. data samples to the left/right side resp.
        rightValIndexList = []
        lclasses = [0]*10   # 0-indexed in Train according to self.trainInputDict["label"]
        rclasses = [0]*10   # lclasses/rclasses store the no. of input samples per class going to their resp. sides
        maxLeft=0
        maxRight=0
        maxLeftClassIndex=-1    # maxLeftClassIndex/maxRightClassIndex stores the index corresponding to class that has the max. samples (among all) that are going to the left/right side resp.  
        maxRightClassIndex=-1
        splitClassIndx = -1     # index of the splitted class (only matters if this option is used above)
        splitClassCnt = -1
        leftSplitClassCnt = 0
        rightSplitClassCnt = 0
        lmaxSplitClassCnt = 0
        rmaxSplitClassCnt = 0
        final_dict = {}
        lTrainDict = {}
        rTrainDict = {}
        lValDict={}
        rValDict={}
        ## resDict = dict(zip(self.trainInputDict["label"].tolist(), modelPrediction.tolist()))

        if options.caseNum==1 or options.caseNum==2:
            maxLeftClasses, maxRightClasses, testCorrectResults, leftClassesToBeRemoved, rightClassesToBeRemoved = self.countBalanceAndThreshold(modelPrediction, labelMap)
            final_dict = self.getSavedFinalSplit()

            # used only when we divide a parent having odd numbered classes into 2 EXACTLY same sized children nodes as was explained earlier
            for k,v in final_dict.items():
                if v == -1:
                    splitClassIndx = k
                    splitClassCnt = (self.trainInputDict["label"].tolist()).count(splitClassIndx)
                    lmaxSplitClassCnt = int((splitClassCnt+1)/2)
                    rmaxSplitClassCnt = int(splitClassCnt/2)
                    break

        for i, val in enumerate(modelPrediction):
            label = self.trainInputDict["label"][i].item()

            if self.isTrain:
                self.trainInputDict["label"][i] = reverseLabelMap[self.trainInputDict["label"][i].item()]

            if val<=0.5:
                if (self.isTrain) and (options.caseNum==1 or options.caseNum==2):
                    #LEFT SIDE
                    ### if not (label in leftClassesToBeRemoved):
                    if (final_dict[label] == 0) or ((final_dict[label] == -1) and leftSplitClassCnt<lmaxSplitClassCnt):
                        lclasses[label]+=1
                        leftChildIndexList.append(i)
                        if (final_dict[label] == -1):
                            leftSplitClassCnt += 1
                        if lclasses[label] > maxLeft:
                            maxLeft = lclasses[label]
                            maxLeftClassIndex = reverseLabelMap[label]
                    #RIGHT SIDE
                    else:
                        rclasses[label]+=1
                        rightChildIndexList.append(i)
                        if (final_dict[label] == -1):
                            rightSplitClassCnt += 1
                        if rclasses[label] > maxRight:
                            maxRight = rclasses[label]
                            maxRightClassIndex = reverseLabelMap[label]
                #LEFT SIDE
                elif (options.caseNum==3) or (not self.isTrain):
                    lclasses[label]+=1
                    leftChildIndexList.append(i)
                    if self.isTrain:
                        if lclasses[label] > maxLeft:
                            maxLeft = lclasses[label]
                            maxLeftClassIndex = reverseLabelMap[label]

            else:
                #RIGHT SIDE
                if (self.isTrain) and (options.caseNum==1 or options.caseNum==2):
                    ### if not (label in rightClassesToBeRemoved):
                    if (final_dict[label] == 1) or ((final_dict[label] == -1) and leftSplitClassCnt>=lmaxSplitClassCnt):
                        rclasses[label]+=1
                        rightChildIndexList.append(i)
                        if (final_dict[label] == -1):
                            rightSplitClassCnt += 1
                        if rclasses[label] > maxRight:
                            maxRight = rclasses[label]
                            maxRightClassIndex = reverseLabelMap[label]
                    #LEFT SIDE
                    else:
                        lclasses[label]+=1
                        leftChildIndexList.append(i)
                        if (final_dict[label] == -1):
                            leftSplitClassCnt += 1
                        if lclasses[label] > maxLeft:
                            maxLeft = lclasses[label]
                            maxLeftClassIndex = reverseLabelMap[label]
                #RIGHT SIDE
                elif (options.caseNum==3) or (not self.isTrain):
                    rclasses[label]+=1
                    rightChildIndexList.append(i)
                    if self.isTrain:
                        if rclasses[label] > maxRight:
                            maxRight = rclasses[label]
                            maxRightClassIndex = reverseLabelMap[label]


        totalLeftImages += float(len(leftChildIndexList))
        totalRightImages += float(len(rightChildIndexList))

        if (not self.isTrain) and (options.probabilistic):     # sending all the parent data to both children nodes
            lTrainDict = {"data":self.trainInputDict["data"], "label":self.trainInputDict["label"], "index":self.trainInputDict["index"]}
            rTrainDict = {"data":self.trainInputDict["data"], "label":self.trainInputDict["label"], "index":self.trainInputDict["index"]}
        else:
            lTrainDict = {"data":self.trainInputDict["data"][leftChildIndexList], "label":self.trainInputDict["label"][leftChildIndexList], "index":self.trainInputDict["index"][leftChildIndexList]}
            rTrainDict = {"data":self.trainInputDict["data"][rightChildIndexList], "label":self.trainInputDict["label"][rightChildIndexList], "index":self.trainInputDict["index"][rightChildIndexList]}

        if options.verbose > 1:
            print("lTrainDict[data].shape: ", lTrainDict["data"].shape, "  lTrainDict[label].shape: ", lTrainDict["label"].shape, "  lTrainDict[index].shape: ", lTrainDict["index"].shape)
            print("rTrainDict[data].shape: ", rTrainDict["data"].shape, "  rTrainDict[label].shape: ", rTrainDict["label"].shape, "  rTrainDict[index].shape: ", rTrainDict["index"].shape)

        # Handling Validation Data
        if self.isTrain:
            total = totalLeftImages + totalRightImages
            if total != 0:
                left_count = (totalLeftImages * len(self.valInputDict["label"])) // (total)
            for i, val in enumerate(self.valInputDict["label"]):
                label = self.valInputDict["label"][i].item()
                
                if options.caseNum==1 or options.caseNum==2:
                    if (final_dict[label] == 0):
                        leftValIndexList.append(i)
                    elif (final_dict[label] == 1):
                        rightValIndexList.append(i)
                
                #TODO: Needs to modified appropriately for this case
                elif options.caseNum==3:
                    if i < left_count:
                        leftValIndexList.append(i)
                    else:
                        rightValIndexList.append(i)

                self.valInputDict["label"][i] = reverseLabelMap[val.item()]


            lValDict = {"data":self.valInputDict["data"][leftValIndexList], "label":self.valInputDict["label"][leftValIndexList], "index":self.valInputDict["index"][leftValIndexList]}
            rValDict = {"data":self.valInputDict["data"][rightValIndexList], "label":self.valInputDict["label"][rightValIndexList], "index":self.valInputDict["index"][rightValIndexList]}
            if options.verbose > 1:
                print("lValDict[data].shape: ", lValDict["data"].shape, "  lValDict[label].shape: ", lValDict["label"].shape, "  lValDict[index].shape: ", lValDict["index"].shape)
                print("rValDict[data].shape: ", rValDict["data"].shape, "  rValDict[label].shape: ", rValDict["label"].shape, "  rValDict[index].shape: ", rValDict["index"].shape)

        giniLeftRatio, giniRightRatio, noOfLeftClasses, noOfRightClasses, giniGain = self.getPredictionAnalysis(totalLeftImages, totalRightImages, lclasses, rclasses)

        leftDataNum = int(totalLeftImages)
        rightDataNum = int(totalRightImages)

        if not (len(leftChildIndexList) == 0):
            maxLeft = float(float(maxLeft)/float(len(leftChildIndexList)))
        if not (len(rightChildIndexList) == 0):
            maxRight = float(float(maxRight)/float(len(rightChildIndexList)))

        # stores all the important/relevant things needed for specifying whether the left and right children nodes are leaf or node and also, their no. of classes and gini gain of the current node resp.  
        handleLeafDict = {"lvl":self.level+1,"noOfLeftClasses":noOfLeftClasses, "noOfRightClasses":noOfRightClasses, "maxLeft":maxLeft, "maxRight":maxRight, "leftDataNum":leftDataNum, "rightDataNum":rightDataNum,"maxLeftClassIndex":maxLeftClassIndex,"maxRightClassIndex":maxRightClassIndex, "giniGain":giniGain}     

        if options.verbose > 0:
            print("RETURNING FROM WORK...")

        if self.isTrain and not self.isLeaf:
            return lTrainDict, lValDict, rTrainDict, rValDict, giniLeftRatio, giniRightRatio, handleLeafDict
        elif not self.isTrain and not self.isLeaf:
            return lTrainDict, rTrainDict, giniLeftRatio, giniRightRatio, noOfLeftClasses, noOfRightClasses, lChildProb, rChildProb
        elif self.isTrain and self.isLeaf:
            return
        else:
            return


    # this is function that runs for testing
    def workTest(self, nodeProb, oneHotTensors, resume=False):
        # if we have a single class in this node which will be given by leafClass atrribute
        # we don't need to fetched trained CNN, as we just have to compare with the class present here 
        if (not self.isTrain) and (not (self.leafClass == -1)):
            x=torch.Tensor(1,1).long()
            x[0] = 1
            est_labels = torch.cat(len(self.trainInputDict["label"].to(self.device))*[x])
            reverseLabelMap = {}
            reverseLabelMap[0] = self.leafClass
            self.checkTestPreds(reverseLabelMap, est_labels, nodeProb, oneHotTensors)

        # used when resume Training is called
        elif (self.isTrain) and (not (self.leafClass == -1)):
            return

        # else we load the CNN model and get the predicted output on the trained CNN
        else:
            reverseLabelMap, labelMap = self.loadCNNModel()

            # used when resume Training is called
            if resume:
                # correctly maps all the inputs as should be done while <make_labels_list()> call in Training
                for i, val in enumerate(self.trainInputDict["label"]):
                    self.trainInputDict["label"][i] = labelMap[val.item()]

                for i, val in enumerate(self.valInputDict["label"]):
                    self.valInputDict["label"][i] = labelMap[val.item()]

            trainInputs = self.trainInputDict["data"]
            numBatches = 5
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

            # getting outputs batch-wise, else GPU goes out of memory :p
            for batch in range(numBatches):
                st_btch, end_btch = batch_sep[batch]
                _, image_next_flat, est_labels, _ = self.cnnModel(self.trainInputDict["data"][st_btch:end_btch].to(self.device))
                if batch == 0:
                    imgNextFlat = image_next_flat.detach().cpu()    
                    estLabels = est_labels.detach().cpu()
                else:
                    imgNextFlat = torch.cat((imgNextFlat,image_next_flat.detach().cpu()))
                    estLabels = torch.cat((estLabels,est_labels.detach().cpu()))

            # if current node is a test node
            if not self.isTrain:
                self.checkTestPreds(reverseLabelMap, estLabels.to(self.device), nodeProb, oneHotTensors)

            # if node is leaf node, we don't need to go further
            if self.isLeaf:
                return

            # else load the Model_to_Train

            if options.optionNum == 1:
                self.loadMLPModel()
            elif options.optionNum == 2:
                self.loadDTModel()
            elif options.optionNum == 3:
                self.loadRFModel()

            # pass the input through Model_to_Train and get the predictions from it
            estLabels = []
            for batch in range(numBatches):    
                st_btch, end_btch = batch_sep[batch]

                if options.optionNum == 1:
                    est_labels = self.mlpModel(imgNextFlat[st_btch:end_btch].to(self.device))
                    est_labels = est_labels.detach().cpu()      # since already a tensor
                elif options.optionNum == 2:
                    est_labels = self.dtModel.predict(imgNextFlat[st_btch:end_btch].to("cpu"))
                    est_labels = torch.from_numpy(est_labels)
                elif options.optionNum == 3:
                    est_labels = self.rfModel.predict(imgNextFlat[st_btch:end_btch].to("cpu"))
                    est_labels = torch.from_numpy(est_labels)
                
                if batch == 0:
                    estLabels = est_labels
                else:
                    estLabels = torch.cat((estLabels, est_labels))

            estLabels = estLabels.view(-1)
            modelPrediction = estLabels.detach()
            modelPrediction = modelPrediction.double()
            rChildProb = modelPrediction.clone()
            lChildProb = 1.0 - rChildProb

            # if probabilistic is used, update lChildProb, rChildProb for its corresponding left and right children nodes
            if options.probabilistic:
                lChildProb = lChildProb*nodeProb
                rChildProb = rChildProb*nodeProb

            modelPrediction += 0.5
            modelPrediction = modelPrediction.long()

            return self.classifyLabels(modelPrediction, reverseLabelMap, labelMap, lChildProb, rChildProb, oneHotTensors)

    
    # this fucntion does 2-means clustering and give the ouput to train Model_to_Train on
    def getTrainPredictionsNotLeaf(self):
        self.loadCNNModel()

        # first we get the output from CNN on the input data
        trainInputs = self.trainInputDict["data"]
        numBatches = 5
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

        if options.verbose > 1:
            print("image_next_flat.shape : ", imgNextFlat.shape)
        img_flat_nmpy = imgNextFlat.numpy()
        countImageTotal = imgNextFlat.shape[0]

        # then we do 2-means clustering on this data
        start = time.time()
        ## kmeans = KMeans(n_clusters=2, n_jobs=-1).fit(img_flat_nmpy)
        kmeans = MiniBatchKMeans(n_clusters=2,random_state=0).fit(img_flat_nmpy)
        end = time.time()
        if options.verbose > 1:
            print("Time Taken by Kmeans is ", end-start)

        # we get the labels from 2-means clustering
        cluster_ids = kmeans.labels_
        if options.verbose > 0:
            print("Kmeans completed successfully...")       

        final_dict = {}

        # build and save final_dict
        if options.caseNum == 1 or options.caseNum == 2:
            # get the correct dictionary to train Model_to_Train on
            leftSortedListOfTuples, rightSortedListOfTuples, expected_dict = self.separateLabels(cluster_ids)
            final_dict = self.makeFinalDict(leftSortedListOfTuples, rightSortedListOfTuples, expected_dict)

        expectedModelLabels = []    # stores the expected labels as per given output by modelToTrain 
        weightVector = []           # stores the weight vector corresponding to each datasample
        countImageRight = 0         
        countImageLeft = 0          # countImageLeft/countImageRight store the total images/samples going left/right resp. 

        # here we are sending data to left and right child directly on the output of KMeans
        if options.caseNum == 3:
            countImageRight = int(np.sum(cluster_ids))
            countImageLeft = countImageTotal - countImageRight

        leftSplitClassCnt = 0
        rightSplitClassCnt = 0
        lmaxSplitClassCnt = 0
        rmaxSplitClassCnt = 0

        if options.caseNum == 1 or options.caseNum == 2:
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

        if options.verbose > 1:
            print("Image Statistics before Model Training : L R : ", countImageLeft, countImageRight)
        minCountImage = min(countImageLeft, countImageRight)
        # construct the expected Model_to_Train labels from the dictionary (with weights so that Model_to_Train is not biased)
        for i in range(len(self.trainInputDict["data"])):
            
            if options.caseNum == 1 or options.caseNum == 2:
                label = self.trainInputDict["label"][i].item()
                if (final_dict[label] == 0) or ((final_dict[label] == -1) and (leftSplitClassCnt<lmaxSplitClassCnt)):
                    expectedModelLabels.append(0)                       # updating expectedModelLabels 
                    weightVector.append(minCountImage/countImageLeft)   # updating weightVector 
                    if (final_dict[label] == -1):
                        leftSplitClassCnt += 1
                else:
                    expectedModelLabels.append(1)
                    weightVector.append(minCountImage/countImageRight)
                    if (final_dict[label] == -1):
                        rightSplitClassCnt += 1

            elif options.caseNum == 3:
                expectedModelLabels.append(cluster_ids[i])
                if cluster_ids[i] == 0:
                    weightVector.append(minCountImage/countImageLeft)
                else:
                    weightVector.append(minCountImage/countImageRight)

        expectedModelLabels = torch.tensor(expectedModelLabels).cpu()
        weightVector = torch.tensor(weightVector, dtype=torch.float32).cpu()
        if options.verbose > 1:
            print("expectedModelLabels.shape : ",expectedModelLabels.shape)
        return imgNextFlat, expectedModelLabels, weightVector


    # function that runs while training
    def workTrain(self):
        # store the parameters of the node in the dictionary
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

        # save it in <options.ckptDir> folder
        torch.save({
                    'nodeDict':nodeDict,
                    }, options.ckptDir+'/node_'+str(self.nodeId)+'.pth')

        # if single class left in training, nothing to do and hence return
        if not (self.leafClass == -1):
            return
        else:
            # else store the correct labels, and construct the label lists for CNN and using that, update the labels of the data samples
            oldData = self.trainInputDict["data"]
            oldLabel = self.trainInputDict["label"]
            labelMap, reverseLabelMap = self.make_labels_list()         

            start = time.time()
            
            # train CNN
            self.cnnModel.to(self.device)
            self.trainCNN(labelMap, reverseLabelMap)
            if options.verbose > 0:
                print("CNN trained successfully...")
            
            end = time.time()
            if options.verbose > 1:
                print("Time Taken during Training CNN: ", end-start, " seconds.")      

            # if node is not a leaf, train ModelToTrain: MLP or DT or RF
            if not self.isLeaf:
                image_next_flat, expectedLabels, weightVector = self.getTrainPredictionsNotLeaf()

                start = time.time()
                if options.optionNum == 1:
                    self.mlpModel.to(self.device)
                    self.trainMLP(image_next_flat, expectedLabels, weightVector)    # train MLP
                    if options.verbose > 0:
                        print("MLP trained successfully...")
                elif options.optionNum == 2:
                    self.trainDT(image_next_flat, expectedLabels, weightVector)      # train DT
                    if options.verbose > 0:
                        print("DT trained successfully...")
                elif options.optionNum == 3:
                    self.trainRF(image_next_flat, expectedLabels, weightVector)      # train RF
                    if options.verbose > 0:
                        print("RF trained successfully...")
                end = time.time()
                if options.verbose > 1:
                    print("Time Taken during Training Model: ", end-start, " seconds.")

            self.trainInputDict["data"] = oldData
            self.trainInputDict["label"] = oldLabel

            # if leaf return
            if self.isLeaf:
                return
            # else do the same thing as work test
            else:
                oneHotTensors = torch.zeros(len(self.trainInputDict["label"]), 10)
                nodeProb = torch.ones(len(self.trainInputDict["label"]))
                return self.workTest(nodeProb,oneHotTensors)
