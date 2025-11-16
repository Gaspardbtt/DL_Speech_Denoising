import torch
import torch.nn as nn


#-------------------------------------------------------------------------------------

# CNN arch for Mask approche

class CNN_MASK(nn.Module):
    def __init__(self):
        super(CNN_MASK, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(8, 1, kernel_size=3, padding=1)  
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.sigmoid(self.conv4(x))
        return x
    

#-------------------------------------------------------------------------------------
