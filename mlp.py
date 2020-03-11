import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class MLP(nn.Module):
    def __init__(self, num_features):
        super(MLP, self).__init__()
        self.fc_layer_1 = nn.Linear(in_features=num_features, out_features=512)
        self.fc_layer_2 = nn.Linear(in_features=512, out_features=256)
        self.fc_layer_3 = nn.Linear(in_features=256, out_features=128)
        self.fc_layer_4 = nn.Linear(in_features=128, out_features=32)
        self.fc_layer_5 = nn.Linear(in_features=32, out_features=8)
        self.fc_layer_6 = nn.Linear(in_features=8, out_features=2)
        self.dropout_1 = nn.Dropout(0.2)
        self.dropout_2 = nn.Dropout(0.2)
        self.dropout_3 = nn.Dropout(0.2)
        self.dropout_4 = nn.Dropout(0.2)
        self.dropout_5 = nn.Dropout(0.2)


    def forward(self, img_flat):
        x = F.relu(self.dropout_1(self.fc_layer_1(img_flat)))
        x = F.relu(self.dropout_2(self.fc_layer_2(x)))
        x = F.relu(self.dropout_3(self.fc_layer_3(x)))
        x = F.relu(self.dropout_4(self.fc_layer_4(x)))
        x = F.relu(self.dropout_5(self.fc_layer_5(x)))
        x = F.relu(self.fc_layer_6(x))
        return x