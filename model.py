import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from kmeans_pytorch import kmeans
from node import *
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
        # self.maxNumberOfNodes = maxNumberOfNodes
    
    def tree_traversal(self):

        # q = Queue(maxsize=self.maxNumberOfNodes)
        queue = []
        nodeNumbers = 1
        newNode = Node(0, nodeNumbers, 10, self.numClasses, self.device, True)
        queue.append(newNode)
        while queue:
            q = queue.pop(0)
            q.work()
            
            

            
            


