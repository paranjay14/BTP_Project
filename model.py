import sys
import torch
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import os.path
from sklearn.model_selection import train_test_split
from new_node import myNode
from sklearn.metrics import confusion_matrix
from getOptions import getOptions
from pptree import *
import random
import time


TOTAL_TRAIN_IMG = 50000     # total training input samples
TOTAL_TEST_IMG = 10000      # total testing input samples
TOTAL_CLASSES = 10          # total no. of different classes present


# Tree class
class Tree:
    # constuctor with default values given in arguments
    def __init__(self, device, maxDepth=1, dominanceThreshold=0.95, classThreshold=1, dataNumThreshold=100, numClasses=TOTAL_CLASSES):
        assert(isinstance(maxDepth,int)) 
        self.maxDepth=maxDepth                         # depth threshold
        assert(isinstance(dominanceThreshold,float)) 
        self.dominanceThreshold=dominanceThreshold     # threshold on class dominance
        assert(isinstance(classThreshold,int)) 
        assert(classThreshold >= 1)
        self.classThreshold=classThreshold             # threshold on number of class in a node
        assert(isinstance(dataNumThreshold,int)) 
        self.dataNumThreshold=dataNumThreshold         # threshold on total input images to a node
        assert(numClasses >= 1)
        assert(isinstance(numClasses,int)) 
        self.numClasses = numClasses                   # total no. of classes spanning the input samples
        self.nodeArray=None                            # stores the list of all the nodes in the tree in a BFS traversal fashion
        self.device = device
        # self.root = None
        # self.maxNumberOfNodes = maxNumberOfNodes
    

    # function to check if any (or both) of the 2 children nodes of some parent node will be leaf nodes or not
    def checkLeafNodes(self, handleLeafDict):
        # boolean for whether left and right children are leaf or not
        isLeafLeft=False
        isLeafRight=False

        # tells whether left or right child is empty child or not
        isemptyNodeLeft=False
        isemptyNodeRight=False

        # this is used to check if leaf has only one class and this further stores the index of that class
        leftLeafClass = -1
        rightLeafClass = -1


        ############# maxDepth #############
        # if the current level of the parent node reaches <maxDepth-1>, then both its children nodes are forcefully made as leaf nodes
        if handleLeafDict["lvl"]==self.maxDepth:
            isLeafLeft=True
            isLeafRight=True
            if options.verbose > 0:
                print("MAX DEPTH REACHED", self.maxDepth)


        ############# dataNumThreshold #############
        # if the total no. of samples in a node are less than the dataNumThreshold, then that node is made a leaf node
        if handleLeafDict["leftDataNum"] <= self.dataNumThreshold:
            isLeafLeft=True
            if options.verbose > 0:
                print("DATANUM THRESHOLD REACHED IN LEFT", handleLeafDict["leftDataNum"], self.dataNumThreshold)
        if handleLeafDict["rightDataNum"] <= self.dataNumThreshold:
            isLeafRight=True        
            if options.verbose > 0:
                print("DATANUM THRESHOLD REACHED IN RIGHT", handleLeafDict["rightDataNum"], self.dataNumThreshold)


        ############# classThreshold #############
        # if a child node contains total no. of classes less than the classThreshold, then it is made a leaf node
        # HANDLE 0, 1 & 2 class cases
        ## LEFT CHILD
        if handleLeafDict["noOfLeftClasses"]==0:
            isemptyNodeLeft=True
            if options.verbose > 0:
                print("CLASS THRESHOLD REACHED 0 IN LEFT", handleLeafDict["noOfLeftClasses"], self.classThreshold)

        elif (self.classThreshold >= 1) and (handleLeafDict["noOfLeftClasses"]==1):
            isLeafLeft=True
            # the only single class is present in a child node is called its leafClass 
            leftLeafClass=handleLeafDict["maxLeftClassIndex"]
            if options.verbose > 0:
                print("CLASS THRESHOLD REACHED 1 IN LEFT", handleLeafDict["noOfLeftClasses"], self.classThreshold, leftLeafClass)

        elif (self.classThreshold >= 2) and (handleLeafDict["noOfLeftClasses"]==2):
            isLeafLeft=True
            if options.verbose > 0:
                print("CLASS THRESHOLD REACHED 2 IN LEFT", handleLeafDict["noOfLeftClasses"], self.classThreshold)

        ## RIGHT CHILD
        if handleLeafDict["noOfRightClasses"]==0:
            isemptyNodeRight=True
            if options.verbose > 0:
                print("CLASS THRESHOLD REACHED 0 IN RIGHT", handleLeafDict["noOfRightClasses"], self.classThreshold)

        elif (self.classThreshold >= 1) and (handleLeafDict["noOfRightClasses"]==1):
            isLeafRight=True
            # the only single class is present in a child node is called its leafClass 
            rightLeafClass=handleLeafDict["maxRightClassIndex"]
            if options.verbose > 0:
                print("CLASS THRESHOLD REACHED 1 IN RIGHT", handleLeafDict["noOfRightClasses"], self.classThreshold, rightLeafClass)

        elif (self.classThreshold >= 2) and (handleLeafDict["noOfRightClasses"]==2):
            isLeafRight=True
            if options.verbose > 0:
                print("CLASS THRESHOLD REACHED 2 IN RIGHT", handleLeafDict["noOfRightClasses"], self.classThreshold)



        ############# dominanceThreshold #############
        # if a child contains a class that has its ratio of input samples to the overall samples in that node greater than dominanceThreshold, then it is made a leaf node
        # also, upon satisfying the criteria, this class is also made the node's leafClass
        if handleLeafDict["maxLeft"] >= self.dominanceThreshold:
            isLeafLeft=True
            leftLeafClass=handleLeafDict["maxLeftClassIndex"]
            if options.verbose > 0:
                print("DOMINANCE THRESHOLD REACHED IN LEFT", handleLeafDict["maxLeft"], self.dominanceThreshold, leftLeafClass)

        if handleLeafDict["maxRight"] >= self.dominanceThreshold:
            isLeafRight=True
            rightLeafClass=handleLeafDict["maxRightClassIndex"]
            if options.verbose > 0:
                print("DOMINANCE THRESHOLD REACHED IN RIGHT", handleLeafDict["maxRight"], self.dominanceThreshold, rightLeafClass)


        return isLeafLeft, isLeafRight, isemptyNodeLeft, isemptyNodeRight, leftLeafClass, rightLeafClass


    # tree traversal while training which, as a result, finally builds and stores the tree including all its nodes' properties and trained models in corresponding files under the <options.ckptDir> directory
    def tree_traversal(self, trainInputDict, valInputDict, resumeTrain, resumeFromNodeId):
        if options.verbose > 0:
            print("\nTRAINING STARTS")
        # make a root node (node definition is given in <new_node.py>) according to max depth given by user
        rootNode = myNode(parentId=0, nodeId=1, device=self.device, isTrain=True, level=0, parentNode=None)
        if self.maxDepth == 0:  # if require only one node in tree
            rootNode.setInput(trainInputDict=trainInputDict, valInputDict=valInputDict, numClasses=self.numClasses, giniValue=0.9, isLeaf=True, leafClass=-1, lchildId=-1, rchildId=-1)
        else:
            rootNode.setInput(trainInputDict=trainInputDict, valInputDict=valInputDict, numClasses=self.numClasses, giniValue=0.9, isLeaf=False, leafClass=-1, lchildId=-1, rchildId=-1)

        # initialising one hot tensors output
        oneHotTensors = torch.zeros(len(trainInputDict["label"]), TOTAL_CLASSES)
        # initialising node probability for each sample by 1
        nodeProb = torch.ones(len(trainInputDict["label"]))

        # node array that stores all the nodes
        self.nodeArray = []
        self.nodeArray.append(rootNode)
        start = 0
        end = 1
        
        # while we haven't traversed the whole node array
        while start != end:
            # get the node at start from list
            node = self.nodeArray[start]
            if options.verbose > 0:
                print("Running nodeId: ", node.nodeId)
            start+=1    # increment start

            # if node is not a leaf node
            # resumfromNodeId helps to start training from a particular node 
            # (did this because - suppose, while training, we got an error. Thus, instead of again starting from the begining root node, we can directly start training from any previously "fully" trained node)
            # (a "fully" trained node is the one whose both - CNN and modelToTrain (if not a leaf node) - models are trained completely) 
            # workTest fuction runs the testing algorithm
            # workTrain function runs the training algorithm
            if not node.isLeaf:
                if (resumeTrain) and (node.nodeId<resumeFromNodeId):
                    lTrainDict, lValDict, rTrainDict, rValDict, giniLeftRatio, giniRightRatio, handleLeafDict = node.workTest(nodeProb, oneHotTensors,True)
                else:
                    lTrainDict, lValDict, rTrainDict, rValDict, giniLeftRatio, giniRightRatio, handleLeafDict = node.workTrain()
            else:
                if (resumeTrain) and (node.nodeId<resumeFromNodeId):
                    node.workTest(nodeProb, oneHotTensors, True)
                else:
                    node.workTrain()

            # if the current node is not leaf node, then it can have children nodes
            if not node.isLeaf:
                # check if any (or both) children node(s) is/are leaf node(s)
                isLeafLeft, isLeafRight, isemptyNodeLeft, isemptyNodeRight, leftLeafClass, rightLeafClass = self.checkLeafNodes(handleLeafDict)
                # loads the current node (i.e. parent) information
                ParentNodeDict = torch.load(options.ckptDir+'/node_'+str(node.nodeId)+'.pth')['nodeDict']

                # if left node is not empty, then create left child according to it being a leaf node or not
                if not isemptyNodeLeft:
                    end += 1
                    lNode = myNode(node.nodeId, end, self.device, True, node.level+1, node)
                    if not isLeafLeft:
                        lNode.setInput(lTrainDict, lValDict, handleLeafDict["noOfLeftClasses"], giniLeftRatio, False, leftLeafClass, -1, -1)
                    else:
                        lNode.setInput(lTrainDict, lValDict, handleLeafDict["noOfLeftClasses"], giniLeftRatio, True, leftLeafClass, -1, -1)
                    # append the child node into array and set the corresponding (left/right) child of the parent node (<node>)
                    self.nodeArray.append(lNode)
                    ParentNodeDict['lchildId'] = lNode.nodeId

                # similarly, we do for the right child node
                if not isemptyNodeRight:
                    end += 1
                    rNode = myNode(node.nodeId, end, self.device, True, node.level+1,node)
                    if not isLeafRight:
                        rNode.setInput(rTrainDict, rValDict, handleLeafDict["noOfRightClasses"], giniRightRatio, False, rightLeafClass, -1, -1)
                    else:
                        rNode.setInput(rTrainDict, rValDict, handleLeafDict["noOfRightClasses"], giniRightRatio, True, rightLeafClass, -1, -1)
                    self.nodeArray.append(rNode)
                    ParentNodeDict['rchildId'] = rNode.nodeId

                # set the gini Gain value for the current node and update its node Dictionary in the file
                ParentNodeDict['giniGain'] = handleLeafDict["giniGain"]
                torch.save({
                    'nodeDict':ParentNodeDict,
                    }, options.ckptDir+'/node_'+str(node.nodeId)+'.pth')


    # this will run on test data
    def testTraversal(self, testInputDict):
        if options.verbose > 0:
            print("\nTESTING STARTS")
        nodeId=1
        
        # load cnn and root node information from checckpoints stored
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
        
        # make root node according to the node stored in checkpoint
        rootNode = myNode(parentId=rootNodeDict['parentId'], nodeId=rootNodeDict['nodeId'], device=self.device, isTrain=False, level=rootNodeDict['level'], parentNode=None)
        rootNode.setInput(trainInputDict=testInputDict, valInputDict={}, numClasses=noOfClasses, giniValue=0.9, isLeaf=isLeafRoot, leafClass=rootNodeDict['leafClass'], lchildId=leftChildId, rchildId=rightChildId)
        
        # initialising one hot tensors output
        oneHotTensors = torch.zeros(len(testInputDict["label"]), TOTAL_CLASSES)
        # initialising node probability for each sample by 1
        nodeProb = torch.ones(len(testInputDict["label"]))

        # stores the test results - corresponding predicted labels and actual labels with their original indices as in original test input
        testPredDict = {}
        testPredDict['actual'] = ((torch.rand(0)).long()).to(self.device)
        testPredDict['pred'] = ((torch.rand(0)).long()).to(self.device)
        testPredDict['index'] = ((torch.rand(0)).long()).to(self.device)
        # saving this dictionary in a file, to be used later in leaf nodes, and also finally while calculating accuracy.
        torch.save({
                    'testPredDict':testPredDict,
                    }, options.ckptDir+'/testPred.pth')

        # this stores level wise accuracy
        LevelDict = {}
        LevelDict['levelAcc'] = {}
        LevelDict['leafAcc'] = [0,0]
        # saving this dictionary in a file, to be used later in each node
        torch.save({
            'levelDict':LevelDict,
            }, options.ckptDir+'/level.pth')

        prevLvl=-1

        # q has list of test nodes which will be travesed in a BFS manner according to the <nodeArray> built while training 
        q = []
        q.append(rootNode)
        start = 0
        end = 1
        while start != end:
            # get node at start index
            node = q[start]
            curLvl=node.level
            # if current level is greater than previous level, marking the start of a new level, update the current leaf accuracy in the file
            if curLvl>prevLvl:
                LevelDict = torch.load(options.ckptDir+'/level.pth')['levelDict']
                LevelDict['levelAcc'][curLvl] = LevelDict['leafAcc'][:]
                torch.save({
                    'levelDict':LevelDict,
                    }, options.ckptDir+'/level.pth')
                prevLvl=curLvl

            start+=1
            # depending on if the node is leaf node or not, we have differnt returns from worktest
            if not node.isLeaf:
                lTrainDict, rTrainDict,  giniLeftRatio, giniRightRatio, noOfLeftClasses, noOfRightClasses, lChildProb, rChildProb = node.workTest(nodeProb, oneHotTensors)
            else:
                node.workTest(nodeProb, oneHotTensors)
            
            # if node is not leaf node, we add the children test nodes using the training checkpoint nodes stored
            if not node.isLeaf:
                # if the corresponding current <node> from training had a left train_child and the left test_child is not empty
                if not ((node.lchildId == -1) or (len(lTrainDict["label"]) == 0)):
                    # loading the node_Dictionary for the left train_child node and using ITS leafClass, level, isLeaf, lchildId & rchildId, 
                    # and setting inputs accordingly for the left test_child node
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
                    # appending the test child node
                    q.append(lNode)
                    end+=1
                
                # similarly, we do for right child
                if not ((node.rchildId == -1) or (len(rTrainDict["label"]) == 0)):
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

                if options.verbose > 1:
                    print ('Nodes sizes = ', noOfLeftClasses, noOfRightClasses)

        # loads the testPred Dictionary
        ckpt = torch.load(options.ckptDir+'/testPred.pth')
        testPredDict = ckpt['testPredDict']
        testPredDict['actual'] = testPredDict['actual'].to("cpu")
        testPredDict['pred'] = testPredDict['pred'].to("cpu")
        ## np.savetxt("testActual.txt", testPredDict['actual'].numpy(), fmt="%d")
        ## np.savetxt("testPred.txt", testPredDict['pred'].numpy(), fmt="%d")

        # build and print the confusion matrix using the predicted and actual labels
        cm = confusion_matrix(testPredDict['actual'], testPredDict['pred'])
        print(cm)
        print()

        # calculate and print the final accuracy obtained
        correct = testPredDict['pred'].eq(testPredDict['actual']).sum().item()
        total = len(testPredDict['actual'])
        if total != 0:
            print('Final Acc: %.3f'% (100.*correct/total))
        else:
            print('Final Acc: 0')
        print()

        # loads the Level Dictionary
        LevelDict = torch.load(options.ckptDir+'/level.pth')['levelDict']
        # calculate and print the level-wise accuracy obtained
        for i,val in enumerate(LevelDict['levelAcc'].items()):
            print('Level %d Acc: %.3f'% (val[0], 100.*val[1][0]/val[1][1]))
        
        # sorts the prediction and actual labels of the samples according to their original indices
        indexList, actList, predList = zip(*sorted(zip(testPredDict["index"], testPredDict['actual'], testPredDict['pred'])))
        predArr = np.array(predList)
        oneHotVector = np.zeros((predArr.size, TOTAL_CLASSES))
        # create a one Hot Vector from the predicted labels and return it
        oneHotVector[np.arange(predArr.size), predArr] = 1      

        return oneHotVector

    
    # function to print tree
    def printTree(self):
        if options.verbose > 0:
            print("\nPRINTING TREE STARTS")
        nodeId=1
        # loads the root node
        rootNodeDict = torch.load(options.ckptDir+'/node_'+str(nodeId)+'.pth')['nodeDict']
        rootNode = myNode(parentId=rootNodeDict['parentId'], nodeId=rootNodeDict['nodeId'], device=self.device, isTrain=False, level=rootNodeDict['level'], parentNode=None)

        # q builds the list of all nodes in a BFS fashion, wherein each parent node stores(ON-THE-GO) the list of all children nodes it has
        # refer init of <myNode>
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
            node.level  =   currNodeDict['level']
            node.leafClass = currNodeDict['leafClass']
            node.numClasses = currNodeDict['numClasses']
            node.numData = currNodeDict['numData']
            node.classLabels = currNodeDict['classLabels']
            node.giniGain = currNodeDict['giniGain']
            node.splitAcc = currNodeDict['splitAcc']
            node.nodeAcc = currNodeDict['nodeAcc']

            if not node.isLeaf:
                if not (node.lchildId == -1):
                    lNode = myNode(node.nodeId, node.lchildId, self.device, False, node.level+1, node)  # appends the current left node in the children of parent node
                    q.append(lNode)
                    end+=1

                if not (node.rchildId == -1):
                    rNode = myNode(node.nodeId, node.rchildId, self.device, False, node.level+1, node)  # appends the current right node in the children of parent node
                    q.append(rNode)
                    end+=1

                
        # <print_tree()> function is used from the online available pptree module (referrence https://github.com/clemtoy/pptree)
        print("nodeId")
        print_tree(rootNode, "children", "nodeId", horizontal=False)    
        print("leafClass") 
        print_tree(rootNode, "children", "leafClass", horizontal=False)     
        print("classLabels")
        print_tree(rootNode, "children", "classLabels", horizontal=True)
        if options.testFlg:
            print("splitAcc")
            print_tree(rootNode, "children", "splitAcc", horizontal=False)
        print("giniGain")
        print_tree(rootNode, "children", "giniGain", horizontal=False)
        print("numData")
        print_tree(rootNode, "children", "numData", horizontal=False)
        if options.testFlg:
            print("nodeAcc")
            print_tree(rootNode, "children", "nodeAcc", horizontal=True)
        ## print("isLeaf") 
        ## print_tree(rootNode, "children", "isLeaf", horizontal=False)
        ## print("numClasses")   
        ## print_tree(rootNode, "children", "numClasses", horizontal=False)
        print() 


    # dfs traversal is called when we are doing probablisitic approach and each data point goes to all the nodes
    def DFS(self, testInputDict):
        if options.verbose > 0:
            print("\nDFS STARTS")
        nodeId=1
        
        # load root node  
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
        
        # initialising one hot tensors output by zeros
        oneHotTensors = torch.zeros(len(testInputDict["label"]), TOTAL_CLASSES)
        # initialising node probability for each sample by 1
        nodeProb = torch.ones(len(testInputDict["label"]))
        
        # call dfs traversal on root node, and hence, from here, it will traverse all the nodes in a DFS fashion
        self.dfsTraversal(rootNode,nodeProb,oneHotTensors)

        # After <dfsTraversal> call returns, oneHotTensors is updated, and using it predicted labels are generated
        _, predicted = oneHotTensors.max(1)
        ## np.savetxt("oneHotTensors.txt", oneHotTensors.numpy(), fmt="%f")
        ## np.savetxt("pred.txt", predicted.numpy(), fmt="%d")
        ## np.savetxt("act.txt", testInputDict["label"].numpy(), fmt="%d")
        
        # calculate and print the final accuracy obtained
        predicted = predicted.to(self.device)
        correct = predicted.eq(testInputDict["label"].to(self.device)).sum().item()
        total = len(oneHotTensors)
        if total != 0:
            print('FINAL Acc: %.3f'% (100.*correct/total))
        else:
            print('FINAL Acc: 0')

        # sorts the prediction and actual labels of the samples according to their original indices
        indexList, actList, predList = zip(*sorted(zip(testInputDict["index"], testInputDict["label"], predicted)))
        predArr = np.array(predList)
        predArr = predArr.astype(int)
        oneHotVector = np.zeros((predArr.size, TOTAL_CLASSES))
        # create a one Hot Vector from the predicted labels and return it
        oneHotVector[np.arange(predArr.size), predArr] = 1      

        return oneHotVector

    # this is dfs helper function
    def dfsTraversal(self, node, nodeProb, oneHotTensors):
        # call testing function of node
        if not node.isLeaf:
            lTrainDict, rTrainDict,  giniLeftRatio, giniRightRatio, noOfLeftClasses, noOfRightClasses, lChildProb, rChildProb = node.workTest(nodeProb, oneHotTensors)
        else:
            node.workTest(nodeProb, oneHotTensors)

        if not node.isLeaf:
            if not ((node.lchildId == -1) or (len(lTrainDict["label"]) == 0)):
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
                
                # this is only difference b/w testTraversal and dfsTraversal, as here the nodes are traversed in a DFS way 
                self.dfsTraversal(lNode, lChildProb, oneHotTensors)
            
            if not ((node.rchildId == -1) or (len(rTrainDict["label"]) == 0)):
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
                self.dfsTraversal(rNode, rChildProb, oneHotTensors)


# loads the train, validation, and test dictionaries
def loadNewDictionaries(seed_val=0, train_num=TOTAL_TRAIN_IMG, test_num=TOTAL_TEST_IMG):
    # sets seed value for the current tree, so that different train_Dictionaries can be created
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)
    torch.backends.cudnn.deterministic=True

    # make directory storing the dataset
    if not os.path.isdir('data/'):
        os.mkdir('data/')

    # make directory storing the various checkoints in the code
    if not os.path.isdir(options.ckptDir+'/'):
        os.mkdir(options.ckptDir+'/')

    if options.verbose > 0:
        print('==> Preparing data...')

    # create the tranformation tobe applied for both train and test samples 
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # getting the training dataset
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    class_labels = trainset.targets

    ## train_batch_sampler = StratifiedSampler(class_labels, TOTAL_TRAIN_IMG)

    # specifies the ratio of the left over image samples other than the training ones 
    test_sz = 1 - (train_num/TOTAL_TRAIN_IMG)
    
    # initialising the validation data and labels as 0 tensor
    valData=torch.empty(0)
    valLabels=torch.empty(0)
    valIndices = torch.arange(0,len(valLabels), step=1, dtype=torch.long)   # stores the corresponding indices for the val. samples
    
    # initialising the train_loader with train_num
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_num, num_workers=0)  # num_workers=0 helps in reproducing the same previously obtained outputs/results

    # if whole TOTAL_TRAIN_IMG are not used as train_num or train images (i.e. < if(train_num!=TOTAL_TRAIN_IMG): >)
    if int(test_sz) != 0:
        # this gets the training and validation indexes from class labels.
        train_idx, valid_idx= train_test_split(
        np.arange(len(class_labels)),
        test_size=test_sz,
        shuffle=True,
        stratify=class_labels)

        # make a train sampler having the indexes we found above
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

        # set the batch size here however required here
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_num, sampler=train_sampler, num_workers=0)
        valid_loader = torch.utils.data.DataLoader(trainset, batch_size=min(100*TOTAL_CLASSES,TOTAL_TRAIN_IMG-train_num), sampler=valid_sampler, num_workers=0)

        '''  -->  PREPEND # FOR ADDING VALIDATION, REMOVE # FOR NO VALIDATION
        ## we have used iterator to get the validation data and labels and store them in dictionary
        vIterator = iter(valid_loader)
        c2 = next(vIterator)
        valData = c2[0].clone().detach()
        valLabels = c2[1].clone().detach()
        valIndices = torch.arange(0,len(valLabels), step=1, dtype=torch.long)
        # '''
        
    # we have used iterator to get train data and labels and store them in dictionary   
    iterator = iter(train_loader)
    c1 = next(iterator)
    trainData = c1[0].clone().detach()
    trainLabels = c1[1].clone().detach()

    # stores the corresponding indices for the train samples
    trainIndices = torch.arange(0,len(trainLabels), step=1, dtype=torch.long)

    # while creating test Input Dict., set the seed to 0 (same for all trees) so that results can be compared and test data remains same for all
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # get the test data set
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_num, shuffle=False, num_workers=0)

    # we have used iterator to get test data and labels and store them in dictionary   
    testIterator = iter(testloader)
    c3 = next(testIterator)
    testData = c3[0].clone().detach()
    testLabels = c3[1].clone().detach()

    # stores the corresponding indices for the test samples
    testIndices = torch.arange(0,len(testLabels), step=1, dtype=torch.long)

    # make dictionaries for train, validation and test data and return them.
    return {"data":trainData, "label":trainLabels, "index":trainIndices}, {"data":valData, "label":valLabels, "index":valIndices}, {"data":testData, "label":testLabels, "index":testIndices}


# prints the final accuracy of the ensemble formed
def getEnsembleAcc(sumOneHotVector, testInputDict):
    predArr = np.argmax(sumOneHotVector, axis=1)
    correct = torch.from_numpy(predArr).eq(testInputDict['label']).sum().item()
    total = len(testInputDict['label'])
    if total != 0:
        print('Ensemble Final Acc: %.3f'% (100.*correct/total))
    else:
        print('Ensemble Final Acc: 0')
    print()


# main function
if __name__ == '__main__':
    # we get the options from arguments (refer <getOptions.py> for all the avaialable options and their default values and execute <python3 model.py --h> for more info.)
    options = getOptions(sys.argv[1:])
    # stores the options in the list
    L = [ "options.trainFlg:" + str(options.trainFlg), " options.testFlg:" + str(options.testFlg), " options.maxDepth:" + str(options.maxDepth),
        " options.ckptDir:" + options.ckptDir, " options.cnnOut:" + str(options.cnnOut),
        " options.mlpFC1:" + str(options.mlpFC1), " options.mlpFC2:" + str(options.mlpFC2),
        " options.cnnLR:" + str(options.cnnLR), " options.mlpLR:" + str(options.mlpLR),
        " options.cnnEpochs:" + str(options.cnnEpochs), " options.mlpEpochs:" + str(options.mlpEpochs),
        " options.cnnSchEpochs:" + str(options.cnnSchEpochs), " options.mlpSchEpochs:" + str(options.mlpSchEpochs),
        " options.cnnSchFactor:" + str(options.cnnSchFactor), " options.mlpSchFactor:" + str(options.mlpSchFactor),
        " options.cnnBatches:" + str(options.cnnBatches), " options.mlpBatches:" + str(options.mlpBatches),
        " options.caseNum:" + str(options.caseNum), " options.optionNum:" + str(options.optionNum),
        " options.ensemble:" + str(options.ensemble), " options.probabilistic:" + str(options.probabilistic),
        " options.verbose:" + str(options.verbose) ]
    
    if options.verbose > 1:
        # printing the options
        print(L)

    start = time.time()

    testInputDict = {}  # initialising test input dictionary
    trainData_size = int(max(TOTAL_TRAIN_IMG/options.ensemble, 100*TOTAL_CLASSES))  # trainData_size stores the no. of training data samples taken per Tree in the Ensemble
    trainData_size -= (trainData_size%TOTAL_CLASSES)
    if options.verbose > 1:
        print("trainData_size:", trainData_size)

    sumOneHotVector = np.zeros((TOTAL_TEST_IMG,TOTAL_CLASSES),dtype=float)  # initialising the Sum of One Hot Vectors of each Tree in the Ensemble
    # concatOneHotVector = np.zeros((TOTAL_TEST_IMG,0),dtype=int)   # initialising the Concat. Output of One Hot Vectors of each Tree in the Ensemble

    # iterating over all the trees in the ensemble
    for treeIndx in range(0, options.ensemble):
        # we load train, validation and test set dictionaries of the given dataset (currently CIFAR-10)
        trainInputDict, valInputDict, testInputDict = loadNewDictionaries(treeIndx, trainData_size) # treeIndx here provides the different seed values so as to create differently trained trees
        if options.verbose > 1:
            print("len(trainInputDict[\"data\"]): ",len(trainInputDict["data"]), ",  len(valInputDict[\"data\"]): ",len(valInputDict["data"]), ",  len(testInputDict[\"data\"]): ",len(testInputDict["data"]))      
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # device = "cuda" if GPU is available, else "cpu"

        ## This below piece of code can be used for debugging by taking those test samples that belong to only a single particular class (classIDToBeTaken)
        ## If using this, then Replace <testInputDict> with <newTestInputDict>
        '''
        newTestInputDict = {}
        newTestInputDict["data"] = torch.rand(0,3,32,32)    # change this according to the dimensions of the input image
        newTestInputDict["label"] = (torch.rand(0)).long()
        newTestInputDict["index"] = (torch.rand(0)).long()
        cnt = 0
        sz = 100    # no. of test samples to be taken
        classIDToBeTaken = 1
        for k,v in enumerate(testInputDict["label"]):
            if (v.item() == classIDToBeTaken) and cnt<sz:
                cnt += 1
                newTestInputDict["data"] = torch.cat((newTestInputDict["data"],testInputDict["data"][k].view(1,3,32,32)),0) # change this according to the dimensions of the input image
                newTestInputDict["label"] = torch.cat((newTestInputDict["label"],v.view(1)),0)
                newTestInputDict["index"] = torch.cat((newTestInputDict["index"],testInputDict["index"][k].view(1)),0)
        if options.verbose > 1:
            print(newTestInputDict["data"].shape, newTestInputDict["label"].shape, newTestInputDict["index"].shape)
        # '''

        # creating the main <tree> object for further processing
        tree = Tree(device, maxDepth=options.maxDepth, dominanceThreshold=0.95, classThreshold = 1, dataNumThreshold = 100, numClasses = TOTAL_CLASSES)

        # if Training Mode is On
        if options.trainFlg:
            '''       ## if require resuming Training from some previously "fully" trained node, then comment this line, it will automatically comment the other below part of code
            resumeFromNodeId = 4
            tree.tree_traversal(trainInputDict, valInputDict, resumeTrain=True, resumeFromNodeId=resumeFromNodeId)
            '''
            resumeFromNodeId = -1   # if freshly training the tree from the root node
            tree.tree_traversal(trainInputDict, valInputDict, resumeTrain=False, resumeFromNodeId=resumeFromNodeId)
            # '''

        # if Testing Mode is On
        if options.testFlg:
            oneHotVector = np.zeros(0)  # stores the one Hot encoding output of the predicted labels Tensor
            # if probabilistic method isn't used
            if not options.probabilistic:
                '''     ## just comment this line in order to use <newTestInputDict>
                oneHotVector = tree.testTraversal(newTestInputDict)     # if the earlier mentioned <newTestInputDict> is created
                '''
                oneHotVector = tree.testTraversal(testInputDict)
                # '''
            # if probabilistic method is used
            else:
                '''     ## just comment this line in order to use <newTestInputDict>
                oneHotVector = tree.DFS(newTestInputDict)   # if the earlier mentioned <newTestInputDict> is created
                '''
                oneHotVector = tree.DFS(testInputDict)
                # '''
            sumOneHotVector += oneHotVector     # summing oneHotTensors of all trees in the ensemble
            # concatOneHotVector = np.concatenate((concatOneHotVector, oneHotVector), axis=1)   # concatenating oneHotTensors of all trees in the ensemble

        if options.trainFlg or options.testFlg:
            tree.printTree()    # printing the built Tree

    if options.testFlg:
        getEnsembleAcc(sumOneHotVector, testInputDict)  # get final accuracy of the ensemble formed

    end = time.time()
    print("Time Taken by whole program is ", float(end-start)/60.0, " minutes.")
