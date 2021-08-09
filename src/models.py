import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import numpy as np


class BSIExtractorModel_Deep_dream(nn.Module):

    def __init__(self,dilation_fac=2):

        super(BSIExtractorModel_Deep_dream, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,20,(40,1),stride=(1,1),padding=0),
            nn.BatchNorm2d(20,momentum=0.1,affine=True,eps=1e-5,),
            nn.Conv2d(in_channels=20,out_channels=80,kernel_size=(1,19),stride=(1,1),),
            nn.BatchNorm2d(80,momentum=0.1,affine=True,eps=1e-5,),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5,1), stride=(5,1)),
            nn.Conv2d(in_channels=80,out_channels=100,kernel_size=(5,1),stride=(1,1),dilation = dilation_fac),
            nn.BatchNorm2d(100,momentum=0.1,affine=True,eps=1e-5,),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5,1), stride=(5,1)),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=100,out_channels=160,kernel_size=(10,1),stride=(1,1),),
            nn.BatchNorm2d(160,momentum=0.1,affine=True,eps=1e-5,),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5,1), stride=(5,1)),
            )

    def forward(self, x):
        fc = self.features(x)
        fc = fc.view(x.shape[0],-1)
        return fc

class BSIExtractorModel(nn.Module):

    def __init__(self,dilation_fac=2):

        super(BSIExtractorModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,20,(40,1),stride=(1,1),padding=0),
            nn.BatchNorm2d(20,momentum=0.1,affine=True,eps=1e-5,),)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=20,out_channels=80,kernel_size=(1,19),stride=(1,1),),
            nn.BatchNorm2d(80,momentum=0.1,affine=True,eps=1e-5,),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5,1), stride=(5,1)),)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=80,out_channels=100,kernel_size=(5,1),stride=(1,1),dilation = dilation_fac),
            nn.BatchNorm2d(100,momentum=0.1,affine=True,eps=1e-5,),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5,1), stride=(5,1)),
            nn.Dropout(p=0.5),)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=100,out_channels=160,kernel_size=(10,1),stride=(1,1),),
            nn.BatchNorm2d(160,momentum=0.1,affine=True,eps=1e-5,),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5,1), stride=(5,1)),)

    def forward(self, x):
        fc = self.convolution_layers(x)
        fc = fc.view(x.shape[0],-1)
        return fc

    def convolution_layers(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class BSIClassifierModel(nn.Module):

    def __init__(self,dilation_fac=2):

        super(BSIClassifierModel, self).__init__()
        self.fc1 = nn.Linear(800,100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100,36)
        self.bn2 = nn.BatchNorm1d(36)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        y_pred = self.fully_connected_layers(x)
        return y_pred

    def fully_connected_layers(self,x):
        x = F.dropout(x, 0.5)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        y_pred = self.logsoftmax(x)
        return y_pred

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)

class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        F1 = 8
        F2 = 16
        D = 2
        kernel_length = 64

        # Layer 1
        self.conv1 = nn.Conv2d(1, 8, (64, 1), stride=1, bias=False, padding=(0, kernel_length// 2))
        self.bn1 = nn.BatchNorm2d(8, momentum=0.01, affine=True, eps=1e-3)

        # Layer 2
        self.conv2 = Conv2dWithConstraint(8, 16, (19, 1), max_norm=1, stride=1, bias=False, groups=8,padding=(0, 0))
        self.bn2 = nn.BatchNorm2d(16, momentum=0.01, affine=True, eps=1e-3)

        self.average_pooling = nn.AvgPool2d(kernel_size=(1, 4))

        self.conv3 =  nn.Conv2d(16,16,(1, 16),stride=1,bias=False,groups=16,padding=(0, 16 // 2))
        self.conv4 = nn.Conv2d(16,16,(1, 1),stride=1,bias=False,padding=(0, 0))
        self.bn3 = nn.BatchNorm2d(16, momentum=0.01, affine=True, eps=1e-3)

        self.average_pooling2 = nn.AvgPool2d(kernel_size=(1, 8))

        # FC Layer
        self.fc1 = nn.Linear(29408, 36)


    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.elu(self.bn2(x))
        x = self.average_pooling(x)
        x = F.dropout(x, 0.5)
        x = self.conv3(x)
        x = self.conv4(x)
        x = F.elu(self.bn3(x))
        x = self.average_pooling2(x)
        x = F.dropout(x, 0.5)

        # FC Layer
        x = x.view(x.shape[0], -1)
        x = F.softmax(self.fc1(x))
        return x


class EEGNet_old(nn.Module):
    def __init__(self):
        super(EEGNet_old, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 19), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints.
        self.fc1 = nn.Linear(496, 36)


    def forward(self, x):
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)

        # FC Layer
        x = x.view(x.shape[0], -1)
        x = F.softmax(self.fc1(x))
        return x



class ShallowNet(nn.Module):
    def __init__(self):
        super(ShallowNet, self).__init__()
        # Layer 1
        self.conv1 = nn.Conv2d(1,40,(25, 1),stride=1)
        self.conv2 = nn.Conv2d(40,40,(1, 19),stride=1,bias=False)
        self.bn1 = nn.BatchNorm2d(40, momentum=0.1,affine = False)

        self.average_pooling = nn.AvgPool2d(kernel_size=(75, 1),stride=(15, 1))


        # FC Layer
        self.fc1 = nn.Linear(2440, 36)

    def square(self,x):
        return torch.mul(x, x)

    def log(self,x):
        return torch.log(x)

    def forward(self, x):

        # Layer 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.square(self.bn1(x))
        x = self.log(self.average_pooling(x))
        x = F.dropout(x, 0.5)

        # FC Layer
        x = x.view(x.shape[0], -1)
        x = F.softmax(self.fc1(x))
        return x


class DeepNet(nn.Module):
    def __init__(self):
        super(DeepNet, self).__init__()
        # Layer 1
        self.conv1 = nn.Conv2d(1,25,(5, 1),stride=1)
        self.conv2 = nn.Conv2d(25,25,(1, 19),stride=(2, 1),bias=False)

        self.bn1 = nn.BatchNorm2d(25, momentum=0.1,affine = False)
        self.pooling1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.conv3 = nn.Conv2d(25,50,(5, 1),stride=(2,1))
        self.bn2 = nn.BatchNorm2d(50, momentum=0.1,affine = False)

        self.conv4 = nn.Conv2d(50,100,(5, 1),stride=(2,1))
        self.bn3 = nn.BatchNorm2d(100, momentum=0.1,affine = False)

        self.conv5 = nn.Conv2d(100,200,(5, 1),stride=(2,1))
        self.bn4 = nn.BatchNorm2d(200, momentum=0.1,affine = False)
        # FC Layer
        self.fc1 = nn.Linear(400, 36)

    def square(self,x):
        return torch.mul(x, x)

    def log(self,x):
        return torch.log(x)

    def forward(self, x):

        # Layer 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.elu(self.bn1(x))
        x = self.pooling1(x)
        x = F.dropout(x, 0.5)

        x = self.conv3(x)
        x = F.elu(self.bn2(x))
        x = self.pooling1(x)
        x = F.dropout(x, 0.5)
        x = self.conv4(x)
        x = F.elu(self.bn3(x))
        x = self.pooling1(x)
        x = F.dropout(x, 0.5)
        x = self.conv5(x)
        x = F.elu(self.bn4(x))
        x = self.pooling1(x)
        x = F.dropout(x, 0.5)
        # FC Layer

        x = x.view(x.shape[0], -1)
        x = F.softmax(self.fc1(x))
        return x
