from __future__ import absolute_import
import torch.nn as nn
import math

window_size = 7

class CNN_constraint(nn.Module):

    def __init__(self, num_classes=2):
        super(CNN_constraint, self).__init__()
        self.conv1 = nn.Conv2d(44, 64, kernel_size=3, stride=1, padding=1) 
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) 
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(128)

        self.max_gap = nn.AdaptiveMaxPool2d(1) 
        self.fc1 = nn.Linear(128, 64) 
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64, num_classes)

        self.term_layer2 = nn.Conv2d(1,1, kernel_size=7,stride=1, padding=0) 


        for m in self.modules(): 
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels 
                m.weight.data.normal_(0, math.sqrt(2. / n)) 
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



    def forward(self, x, x_term, x_term1, x_term2):
        
        x = self.conv1(x)  
        x = self.relu1(x) 
        x = self.bn1(x)
        x = x * x_term 
        x = self.conv2(x)  
        x = self.relu2(x) 
        x = self.bn2(x)     

        x = self.max_gap(x)
        x = x.view(x.size(0), -1) 
        
        x_term1 = x_term1[:,:,math.floor(window_size/2),math.floor(window_size/2)] 
        x = x * x_term1
        x = self.fc1(x)
        x = self.relu3(x)
        x = nn.Dropout()(x) 
        x = self.fc2(x) 
      
        x_term2 = self.term_layer2(x_term2)
        x_term2 = nn.Sigmoid()(x_term2)

        return x, x_term2



