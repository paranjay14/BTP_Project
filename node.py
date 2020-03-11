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
    # def __init__(self, inputs, targets, class_bit, parent, node_num, num_classes, device):
    #     self.parent = parent
    #     self.node_num = node_num
    #     self.num_classes = num_classes
    #     self.device = device
    #     self.is_train = True
    #     self.get_input(inputs, targets, class_bit)
        
    # def __init__(self, parent, node_num, num_classes, device, is_train=True):
    #     self.class_bit = class_bit
    #     self.num_images = 0
    #     self.images = None
    #     self.labels = None
    #     self.parent = parent
    #     self.node_num = node_num
    #     self.device = device
    #     self.num_classes = num_classes
    #     self.is_train = is_train
    #     self.isDecidingNode = False
    #     self.cnn_model = None
    #     self.mlp_model = None
    
    def __init__(self, images, labels, class_bit, parent, node_num, num_classes, device, is_train=True):
        self.class_bit = class_bit
        img_size = images.shape[2]
        self.num_images = images.shape[0]
        self.images = images
        self.labels = labels
        self.parent = parent
        self.node_num = node_num
        self.device = device
        self.num_classes = num_classes
        self.is_train = is_train
        self.cnn_model = CNN(num_class=num_classes, img_size=img_size, kernel=3)
        num_features = (img_size-2)*(img_size-2)*3
        self.mlp_model = MLP(num_features)


    def get_input(self, images, labels, class_bit):
        self.class_bit = class_bit
        self.images = images
        self.labels = labels
        img_size = images.shape[2]
        num_features = (img_size-2)*(img_size-2)*3
        self.num_images = images.shape[0]
        self.cnn_model = CNN(num_class=self.num_classes, img_size=img_size, kernel=3)
        self.mlp_model = MLP(num_features)
    
    def train_cnn(self, end_epoch):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.cnn_model.parameters(),lr=0.001)
        self.images = self.images.to(self.device)
        self.labels = self.labels.to(self.device)
        targets = self.labels
        inputs = self.images
        for epoch in range(end_epoch):
            train_loss = 0
            correct = 0
            total = 0
            
            optimizer.zero_grad()
            _, _, est_labels = self.cnn_model(inputs)
            batch_loss = loss_fn(est_labels, targets)
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item()
            _, predicted = est_labels.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print(epoch, 'Loss: %.3f | Acc: %.3f'% (train_loss, 100.*correct/total))

        torch.save({
                    'epoch':epoch,
                    'model_state_dict':self.cnn_model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'train_loss':train_loss,
                    }, 'ckpt/node_cnn_'+str(self.parent)+'_'+str(self.node_num)+'.pth')


    # def validation_test(self, val)

    def train_mlp(self, end_epoch, inputs, targets):
        loss_fn = nn.CrossEntropyLoss()
        # loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(self.mlp_model.parameters(),lr=0.001)
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        for epoch in range(end_epoch):
            train_loss = 0
            correct = 0
            total = 0
            
            optimizer.zero_grad()
            est_labels = self.mlp_model(inputs)
            batch_loss = loss_fn(est_labels, targets)
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item()
            _, predicted = est_labels.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print(epoch, 'Loss: %.3f | Acc: %.3f'% (train_loss, 100.*correct/total))

        torch.save({
                    'epoch':epoch,
                    'model_state_dict':self.mlp_model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'train_loss':train_loss,
                    }, 'ckpt/node_mlp_'+str(self.parent)+'_'+str(self.node_num)+'.pth')



    def work(self):
        self.cnn_model.to(self.device)
        self.mlp_model.to(self.device)
        if self.is_train:
            # end_epoch = 1
            end_epoch = 200
            self.cnn_model.train()
            self.train_cnn(end_epoch)

        ckpt = torch.load('ckpt/node_cnn_'+str(self.parent)+'_'+str(self.node_num)+'.pth')
        self.cnn_model.load_state_dict(ckpt['model_state_dict'])
        self.cnn_model.eval()
        # self.cnn_model.to(device)

        image_next, image_next_flat, _ = self.cnn_model(self.images)
        image_next = image_next.detach()
        image_next_flat = image_next_flat.detach()
        cluster_ids, _ = kmeans_output(image_next_flat, self.device)

        l = {}
        r = {}
        for i in range(10):
            l[i] = 0
            r[i] = 0
        for i in range(self.num_images):
            label = self.labels[i].item()
            if cluster_ids[i] == 0:
                l[label]+=1
            else:
                r[label]+=1

        final_dict = {}
        for i in range(10):
            if self.class_bit[i]:
                if l[i]>=r[i]:
                    # final_dict[i] = [1.0,0.0]
                    final_dict[i] = 0
                else:
                    # final_dict[i] = [0.0,1.0]
                    final_dict[i] = 1

        actual_cluster_ids = []
        for i in range(self.num_images):
            label = self.labels[i].item()
            actual_cluster_ids.append(final_dict[label])
        actual_cluster_ids = torch.tensor(actual_cluster_ids, device=self.device)
        print(actual_cluster_ids.shape)

        if self.is_train:
            end_epoch = 40
            # end_epoch = 1
            self.mlp_model.train()
            self.train_mlp(end_epoch, image_next_flat, actual_cluster_ids)

        ckpt = torch.load('ckpt/node_mlp_'+str(self.parent)+'_'+str(self.node_num)+'.pth')
        self.mlp_model.load_state_dict(ckpt['model_state_dict'])
        self.mlp_model.eval()

        # inputs = inputs.to(self.device)
        # print(image_next_flat)
        est_labels = self.mlp_model(image_next_flat)
        print("fsfsafsfsaf")
        # print(est_labels[0])
        # print(est_labels[1])
        _, predicted = est_labels.max(1)
        # print(predicted)
        limages = []
        rimages = []
        llabels = []
        rlabels = []
        lclasses = [0]*10
        rclasses = [0]*10
        for i, val in enumerate(predicted):
            if val==0:
                limages.append((image_next[i].detach()).tolist())
                lclasses[self.labels[i].item()]+=1
                llabels.append(self.labels[i].item())
            else:
                rimages.append((image_next[i].detach()).tolist())
                rclasses[self.labels[i].item()]+=1
                rlabels.append(self.labels[i].item())

        print(lclasses)
        print(rclasses)
        # print(len(limages))
        # print(len(rimages))
        print("ASfsa")
        ltensor = torch.tensor(limages)
        rtensor = torch.tensor(rimages)
        

    # def test(self):
    #     ckpt = torch.load('ckpt/node_mlp_'+str(self.parent)+'_'+str(self.node_num)+'.pth')
    #     self.mlp_model.load_state_dict(ckpt['model_state_dict'])
    #     self.mlp_model.eval()

    #     # inputs = inputs.to(self.device)
    #     est_labels = self.mlp_model(image_next_flat)
    #     print(est_labels)

    # def predict_next_node_class(self, ):


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(2352).to(device)

    summary(model, input_size=(2352,), batch_size=128)
    
        
