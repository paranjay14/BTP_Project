import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class CNN(nn.Module):
    def __init__(self,img_size=32, in_channels=3, out_channels=3, kernel=5, batch_size=1000, num_class=10, extra_error=False):
        super(CNN, self).__init__()
        self.extra_error = extra_error
        self.num_class = num_class
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=1, padding=0)
        self.conv2 = nn.ConvTranspose2d(in_channels=out_channels, out_channels=in_channels, kernel_size=kernel, stride=1, padding=0)
        
        # self.softmax = nn.Softmax(dim=10) 
        self.features = (img_size-kernel+1)*(img_size-kernel+1)*out_channels
        self.fc_layer = nn.Linear(in_features=self.features, out_features=num_class)

    def forward(self, img):
        out = F.relu(self.conv1(img))
        f_same = F.relu(self.conv2(out))

        flat_out = out.view(-1, self.features)
        final = self.fc_layer(flat_out)

        if self.extra_error:
            return out, flat_out, final
        else:
            return out, flat_out, final, f_same
            