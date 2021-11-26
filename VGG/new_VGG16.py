import torch as th
import torch.nn as nn
from vgg_network import Network
from vgg_apply import PruningMethod
from torchsummary import summary
import math
from collections import OrderedDict

class VGG16(nn.Module,Network,PruningMethod):
    def __init__(self, n_c, a_type, n_iterations):
        super(VGG16, self).__init__()

        self.a_type = a_type

        if a_type == 'relu':
            self.activation = nn.ReLU()
        elif a_type == 'tanh':
            self.activation = nn.Tanh()
        elif a_type == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif a_type == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            print('Not implemented')
            raise

        # First encoder
        self.layer1 = nn.Sequential(
                *([nn.Conv2d(3, 64, kernel_size=3, padding=1),
                   nn.BatchNorm2d(64),
                   self.activation]))
        self.layer2 = nn.Sequential(
                *([nn.Conv2d(64, 64, kernel_size=3, padding=1),
                   nn.BatchNorm2d(64),
                   self.activation]))
        # Second encoder
        self.layer3 = nn.Sequential(
                *([nn.Conv2d(64, 128, kernel_size=3, padding=1),
                   nn.BatchNorm2d(128),
                   self.activation]))
        self.layer4 = nn.Sequential(
                *([nn.Conv2d(128, 128, kernel_size=3, padding=1),
                   nn.BatchNorm2d(128),
                   self.activation]))

        # Third encoder
        self.layer5 = nn.Sequential(
                *([nn.Conv2d(128, 256, kernel_size=3, padding=1),
                   nn.BatchNorm2d(256),
                   self.activation]))
        self.layer6 = nn.Sequential(
                *([nn.Conv2d(256, 256, kernel_size=3, padding=1),
                   nn.BatchNorm2d(256),
                   self.activation]))
        self.layer7 = nn.Sequential(
                *([nn.Conv2d(256, 256, kernel_size=3, padding=1),
                   nn.BatchNorm2d(256),
                   self.activation]))

        # Fourth encoder
        self.layer8 = nn.Sequential(
                *([nn.Conv2d(256, 512, kernel_size=3, padding=1),
                   nn.BatchNorm2d(512),
                   self.activation]))
        self.layer9 = nn.Sequential(
                *([nn.Conv2d(512, 512, kernel_size=3, padding=1),
                   nn.BatchNorm2d(512),
                   self.activation]))
        self.layer10 = nn.Sequential(
                *([nn.Conv2d(512, 512, kernel_size=3, padding=1),
                   nn.BatchNorm2d(512),
                   self.activation]))

        # Fifth encoder
        self.layer11 = nn.Sequential(
                *([nn.Conv2d(512, 512, kernel_size=3, padding=1),
                   nn.BatchNorm2d(512),
                   self.activation]))
        self.layer12 = nn.Sequential(
                *([nn.Conv2d(512, 512, kernel_size=3, padding=1),
                   nn.BatchNorm2d(512),
                   self.activation]))
        self.layer13 = nn.Sequential(
                *([nn.Conv2d(512, 512, kernel_size=3, padding=1),
                   nn.BatchNorm2d(512),
                   self.activation]))

        # Classifier
        self.fc1 = nn.Sequential(*([
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                self.activation]))
        #self.fc2 = nn.Sequential(*([
        #        nn.Dropout(),
        #        nn.Linear(4096, 4096),
        #        self.activation]))
        self.classifier = nn.Sequential(
                *([nn.Linear(512, n_c),]))

        for m in self.modules():
            self.weight_init(m)

        self.pool = nn.MaxPool2d(2, 2)
        self.softmax = nn.Softmax(dim=1)
        
       
        self.sigmas = th.zeros((15, n_iterations)).cuda()
        self.MI = th.zeros((n_iterations, 15,2)).cuda()
        self.sigmas_fixed=th.zeros((15))
        self.test_accuracy=[]
        self.layer_name_num={}
        self.pruned_filters={}
        self.remaining_filters={}
        self.remaining_filters_each_epoch=[]
        self.train_accuracy=[]
        self.trainable_parameters=[]
        self.flops=[]
        self.sub_MI_X=[]
        self.sub_MI_Y=[]

    def forward(self, x):

        # Encoder 2
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        pool1 = self.pool(layer2)

        # Encoder 2
        layer3 = self.layer3(pool1)
        layer4 = self.layer4(layer3)
        pool2 = self.pool(layer4)

        # Encoder 3
        layer5 = self.layer5(pool2)
        layer6 = self.layer6(layer5)
        layer7 = self.layer7(layer6)
        pool3 = self.pool(layer7)

        # Encoder 4
        layer8 = self.layer8(pool3)
        layer9 = self.layer9(layer8)
        layer10 = self.layer10(layer9)
        pool4 = self.pool(layer10)

        # Encoder 5
        layer11 = self.layer11(pool4)
        layer12 = self.layer12(layer11)
        layer13 = self.layer13(layer12)
        #pool5 = self.pool(layer13)

        #avgpool 
        avg_x=nn.AvgPool2d(2)(layer13)

        # Classifier
        fc1 = self.fc1(avg_x.view(avg_x.size(0), -1))
        
        classifier = self.classifier(fc1)
        return [classifier, fc1, layer13, layer12, layer11,layer10,
         layer9, layer8, layer7, layer6, layer5,layer4,
          layer3, layer2, layer1]



