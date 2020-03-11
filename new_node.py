import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn import CNN
from mlp import MLP
from torchsummary import summary
from kmeans_pytorch import kmeans


def kmeans_output(all_images_flat, device, num_clusters=2):
    cluster_ids_x, cluster_centers = kmeans(X=all_images_flat, num_clusters=num_clusters, distance='euclidean', device=device)
    return cluster_ids_x, cluster_centers


class Node:
    def __init__(self, parentId, nodeId, device, isTrain):
        self.parentId = parentId
        self.nodeId = nodeId
        # self.numClasses = numClasses
        self.device = device
        self.isTrain = isTrain
        
    def setInput(self, trainInputDict, valInputDict, numClasses):
        self.trainInputDict = trainInputDict
        self.valInputDict = valInputDict
        imgSize = trainInputDict["data"][0].shape[2]
        inChannels = trainInputDict["data"][0].shape[0]
        print(imgSize)
        outChannels = 16
        kernel = 5
        self.cnnModel = CNN(img_size=imgSize, in_channels=inChannels, out_channels=outChannels, num_class=numClasses, kernel=kernel)
        numFeatures = self.cnnModel.features
        self.mlpModel = MLP(numFeatures)
        # self.numEpochs = numEpochs
        self.numClasses = numClasses
        # self.mlpModel = MLP()


    def trainCNN(self):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.cnnModel.parameters(),lr=0.001)
        # print
        trainLabels = self.trainInputDict["label"]
        trainInputs = self.trainInputDict["data"]
        trainLabels = trainLabels.to(self.device)
        trainInputs = trainInputs.to(self.device)

        numBatches = 10
        batchSize = int((len(trainInputs) + numBatches)/numBatches)
        numEpochs = 5
        for epoch in range(numEpochs):
            st_btch = 0
            total = 0
            correct = 0
            train_loss = 0
            self.cnnModel.train()
            for batch in range(numBatches):
                end_btch = min(st_btch + batchSize, len(trainInputs))

                optimizer.zero_grad()
                _, _, est_labels = self.cnnModel(trainInputs[st_btch:end_btch])
                batch_loss = loss_fn(est_labels, trainLabels[st_btch:end_btch])
                batch_loss.backward()
                optimizer.step()

                train_loss += batch_loss.item()
                _, predicted = est_labels.max(1)
                # total += trainLabels.size(0)
                total += end_btch - st_btch
                correct += predicted.eq(trainLabels[st_btch:end_btch]).sum().item()
                st_btch = end_btch

            #TODO: Add validation iteration here(first change mode to eval)
            
            print(epoch, 'Train Loss: %.3f | Train Acc: %.3f'% (train_loss, 100.*correct/total))

            torch.save({
                    'epoch':epoch,
                    'model_state_dict':self.cnnModel.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'train_loss':train_loss,
                    }, 'ckpt/node_cnn_'+str(self.parentId)+'_'+str(self.nodeId)+'.pth')


    def trainMLP(self, trainInputs, trainTargets):
        loss_fn = nn.CrossEntropyLoss()
        # loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(self.mlpModel.parameters(),lr=0.001)
        trainInputs = trainInputs.to(self.device)
        trainTargets = trainTargets.to(self.device)

        numBatches = 10
        batchSize = int(len(trainInputs) + numBatches/numBatches)
        numEpochs = 5
        for epoch in range(numEpochs):
            st_btch = 0
            train_loss = 0
            correct = 0
            total = 0
            self.mlpModel.train()

            for batch in range(numBatches):    
                end_btch = min(st_btch + batchSize, len(trainInputs))

                optimizer.zero_grad()
                est_labels = self.mlpModel(trainInputs[st_btch:end_btch])
                batch_loss = loss_fn(est_labels, trainTargets[st_btch:end_btch])
                batch_loss.backward()
                optimizer.step()

                train_loss += batch_loss.item()
                _, predicted = est_labels.max(1)
                # total += trainTargets.size(0)
                total += end_btch - st_btch
                correct += predicted.eq(trainTargets[st_btch:end_btch]).sum().item()

            #TODO: add validation testing of MLP here
            
            print(epoch, 'Loss: %.3f | Acc: %.3f'% (train_loss, 100.*correct/total))

            torch.save({
                    'epoch':epoch,
                    'model_state_dict':self.mlpModel.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'train_loss':train_loss,
                    }, 'ckpt/node_mlp_'+str(self.parentId)+'_'+str(self.nodeId)+'.pth')

    
    def work(self):
        self.cnnModel.to(self.device)
        self.mlpModel.to(self.device)
        if self.isTrain:
            self.trainCNN()
        
        ckpt = torch.load('ckpt/node_cnn_'+str(self.parentId)+'_'+str(self.nodeId)+'.pth')
        self.cnnModel.load_state_dict(ckpt['model_state_dict'])
        self.cnnModel.eval()

        image_next, image_next_flat, _ = self.cnnModel(self.trainInputDict["data"])
        image_next = image_next.detach()
        image_next_flat = image_next_flat.detach()
        cluster_ids, _ = kmeans_output(image_next_flat, self.device)

        #TODO: do 

        leftCnt = {}
        rightCnt = {}
        for i in range(self.numClasses):
            leftCnt[i] = 0
            rightCnt[i] = 0
        for i in range(len(self.trainInputDict["data"])):
            label = self.trainInputDict["label"][i].item()
            if cluster_ids[i] == 0:
                leftCnt[label]+=1
            else:
                rightCnt[label]+=1

        final_dict = {}
        for i in range(self.numClasses):
            if leftCnt[i]>=rightCnt[i]:
                # final_dict[i] = [1.0,0.0]
                final_dict[i] = 0
            else:
                # final_dict[i] = [0.0,1.0]
                final_dict[i] = 1

        #TODO: separate for validation set too

        expectedMlpLabels = []
        for i in range(len(self.trainInputDict["data"])):
            label = self.trainInputDict["label"][i].item()
            expectedMlpLabels.append(final_dict[label])
        expectedMlpLabels = torch.tensor(expectedMlpLabels, device=self.device)
        print(expectedMlpLabels.shape)
        
        if self.isTrain:
            self.trainMLP(image_next_flat, expectedMlpLabels)

        ckpt = torch.load('ckpt/node_mlp_'+str(self.parentId)+'_'+str(self.nodeId)+'.pth')
        self.mlpModel.load_state_dict(ckpt['model_state_dict'])
        self.mlpModel.eval()

        est_labels = self.mlpModel(image_next_flat)
        _, mlpPrediction = est_labels.max(1)

        trainLimages = []
        trainRimages = []
        trainLLabels = []
        trainRLabels = []
        lclasses = [0]*10
        rclasses = [0]*10
        for i, val in enumerate(mlpPrediction):
            if val==0:
                trainLimages.append((image_next[i].detach()).tolist())
                lclasses[self.trainInputDict["label"][i].item()]+=1
                trainLLabels.append(self.trainInputDict["label"][i].item())
            else:
                trainRimages.append((image_next[i].detach()).tolist())
                rclasses[self.trainInputDict["label"][i].item()]+=1
                trainRLabels.append(self.trainInputDict["label"][i].item())

        lTrainDict = {"data":torch.tensor(trainLimages), "label":torch.tensor(trainLLabels)}
        rTrainDict = {"data":torch.tensor(trainRimages), "label":torch.tensor(trainRLabels)}

        lValDict = {}
        rValDict = {}
        #TODO: populate validation dictionaries too
        return lTrainDict, lValDict, rTrainDict, rValDict

        