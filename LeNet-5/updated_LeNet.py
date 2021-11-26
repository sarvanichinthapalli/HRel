import torch as th
import torch.nn as nn
from updated_network import Network
import torch.nn.functional as F
from updated_apply import PruningMethod

class LeNet(nn.Module, Network,PruningMethod):
    def __init__(self, n_iterations):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(50 * 4 * 4, 800)  # 5x5 image dimension
        self.fc2 = nn.Linear(800, 500)
        self.fc3 = nn.Linear(500, 10)
        self.a_type='relu'
        for m in self.modules():
            self.weight_init(m)

        #self.pool_layer = nn.MaxPool2d(2, 2)
        #self.softmax = nn.Softmax(dim=1)

        self.sigmas = th.zeros((5, n_iterations)).cuda()
        self.MI = th.zeros((n_iterations, 5,2)).cuda()
        
        self.sigmas_fixed=[0,0]

        self.layer_name_num={}
        self.pruned_filters={}
        self.remaining_filters={}

        self.remaining_filters_each_epoch=[]
        self.test_accuracy=[]
        self.train_accuracy=[]

        self.train_loss=[]
        self.test_loss=[]

        self.trainable_parameters=[]
        self.flops=[]
        self.sub_MI_X=[]
        self.sub_MI_Y=[]


    def forward(self, x):
        layer1 = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        layer2 = F.max_pool2d(F.relu(self.conv2(layer1)), 2)
        layer2_p = layer2.view(-1, int(layer2.nelement() / layer2.shape[0]))
        layer3= F.relu(self.fc1(layer2_p))
        layer4 = F.relu(self.fc2(layer3))
        layer5 = self.fc3(layer4)
        return layer5


