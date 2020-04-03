## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # input image size 1 x 224 x 224
        self.conv1 = nn.Conv2d(1, 32, 5)
        # Now 32 x 220 x 220    (W-F)/S + 1  = (224-5)/1 + 1 = 220
        I.xavier_normal_(self.conv1.weight)
        self.pool1 = nn.MaxPool2d(2,2)
        self.drop2d1 = nn.Dropout2d(0.1)
        # Now 32 x 110 x 110 
        
        
        self.conv2 = nn.Conv2d(32, 64, 4)
        # Now 64 x 107 x 107  (W-F)/S + 1 = (110-4)/1 + 1 = 107
        I.xavier_normal_(self.conv2.weight)
        self.pool2 = nn.MaxPool2d(2,2)
        self.drop2d2 = nn.Dropout2d(0.2)
        #Now 64 x 53 x 53
        
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        # Now 128 x 51 x 51   (53-3)/S + 1 = 51
        I.xavier_normal_(self.conv3.weight)
        self.pool3 = nn.MaxPool2d(2,2)
        # Now 128 x 25 x 25
        self.drop2d3 = nn.Dropout2d(0.3)
        
       
        self.conv4 = nn.Conv2d(128, 256, 2)
        # Now 256 x 24 x 24 (25-2)/1 + 1 = 24
        I.xavier_normal_(self.conv4.weight)
        self.pool4 = nn.MaxPool2d(2,2)
        self.drop2d4 = nn.Dropout2d(0.4)
        # Now 256 x 12 x 12
        
        self.conv5 = nn.Conv2d(256, 512, 2)
        # Now 512 x 11 x 11 (12-2)/1 + 1 = 11
        I.xavier_normal_(self.conv5.weight)
        self.pool5 = nn.MaxPool2d(2,2)
        # Now 512 x 5 x 5 
        self.drop2d5 = nn.Dropout2d(0.5)
        
        self.fc1 = nn.Linear(512*5*5, 6000)
        I.xavier_normal_(self.fc1.weight)
        self.bn1 = nn.BatchNorm1d(6000)
        self.drop1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(6000, 3000)
        I.xavier_normal_(self.fc2.weight)
        self.bn2 = nn.BatchNorm1d(3000)
        self.drop2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(3000, 1000)
        I.xavier_normal_(self.fc3.weight)
        self.drop3 = nn.Dropout(0.5)
     
        self.fc4 = nn.Linear(1000, 136)
        I.xavier_normal_(self.fc4.weight)
        # output size 68 x 2
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.drop2d1(self.pool1(F.leaky_relu(self.conv1(x), 0.2)))
        
        x = self.drop2d2(self.pool2(F.leaky_relu(self.conv2(x), 0.2)))
        
        x = self.drop2d3(self.pool3(F.leaky_relu(self.conv3(x), 0.2)))
        
        x = self.drop2d4(self.pool4(F.leaky_relu(self.conv4(x), 0.2)))
        
        x = self.drop2d5(self.pool5(F.leaky_relu(self.conv5(x), 0.2)))
        #x = self.drop2d6(self.pool6(F.relu(self.bn6(self.conv6(x)))))
        
        x = x.view(x.size(0), -1)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.drop3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
