import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class MLP(nn.Module):
    def __init__(self, num_features, use_bn=False):
        super(MLP, self).__init__()
        if use_bn:
            self.fc_layer_1 = nn.Sequential(
                                    nn.Linear(in_features=num_features, out_features=512),
                                    nn.BatchNorm1d(512),
                              )
            self.fc_layer_2 = nn.Sequential(
                                    nn.Linear(in_features=512, out_features=16),
                                    nn.BatchNorm1d(16),
                              )
        else:
            self.fc_layer_1 = nn.Linear(in_features=num_features, out_features=512) # remember to check this when num_features decrease!!
            self.fc_layer_2 = nn.Linear(in_features=512, out_features=16)
        self.fc_layer_3 = nn.Linear(in_features=16, out_features=1)
        self.dropout_1 = nn.Dropout(0.4)
        self.dropout_2 = nn.Dropout(0.3)


    def forward(self, img_flat):
        x = F.leaky_relu(self.dropout_1(self.fc_layer_1(img_flat)), negative_slope = 0.2)
        x = F.leaky_relu(self.dropout_2(self.fc_layer_2(x)), negative_slope = 0.2)
        x = torch.sigmoid(self.fc_layer_3(x))
        return x