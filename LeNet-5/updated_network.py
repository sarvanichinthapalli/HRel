
import torch as th
import torch.nn as nn
import torch.nn.init as init
#import pandas as pd
import numpy as np

import csv 
from time import localtime, strftime
import os 

seed = 1787
#random.seed(seed)
#import os
#os.environ['PYTHONHASHSEED'] = str(seed)

#th.manual_seed(seed)
#th.cuda.manual_seed(seed)
#th.cuda.manual_seed_all(seed)
#th.backends.cudnn.deterministic = True


class Network():


    def dist_mat(self, x):

        try:
            x = th.from_numpy(x)
        except TypeError:
            x = x

        if len(x.size()) == 4:
            x = x.view(x.size()[0], -1)
        dist = th.norm(x[:, None] - x, dim=2)
        return dist

    def entropy(self, *args):

        for idx, val in enumerate(args):
            if idx == 0:
                k = val.clone()
            else:
                k *= val

        k /= k.trace()

        c = th.tensor([0]).cuda()
        #print(not(th.symeig(k)[0] < c).any())

        eigv = th.abs(th.symeig(k, eigenvectors=False)[0])
        temp=eigv.clone()
        eigv_log2= temp.log2().cuda()

        if((eigv==c).any()):

          zero_indices=(eigv == 0).nonzero().tolist()
          #small=th.tensor([0.999999999]).cuda()
          small=th.tensor([0.0000000099]).cuda()
          small_value=small.detach().clone()
          for i in zero_indices:
            eigv_log2[i]=small_value
          #print(eigv*eigv_log2.sum())

        return -(eigv*(eigv_log2)).sum()

    def kernel_mat(self, x, k_y, sigma=None, epoch=None, idx=None):

        d = self.dist_mat(x)
        #print('ready for sigma calculation',epoch,sigma)
        if sigma is None:
            if epoch > 20:
                sigma_vals = th.linspace(0.3, 10*d.mean(), 100).cuda() 
                #sigma_vals = th.linspace(0.1*d.mean(), 10*d.mean(), 50).cuda() 
              
            else:
                sigma_vals = th.linspace(0.3, 10*d.mean(), 300).cuda()
                #sigma_vals = th.linspace(0.1*d.mean(), 10*d.mean(), 75).cuda()
            L = []
            for sig in sigma_vals:
                k_l = th.exp(-d ** 2 / (sig ** 2)) / d.size(0)
                L.append(self.kernel_loss(k_y, k_l))

            if epoch == 0:
                self.sigmas[idx, epoch] = sigma_vals[L.index(max(L))]
            else:
                self.sigmas[idx, epoch] = 0.9*self.sigmas[idx, epoch-1] + 0.1*sigma_vals[L.index(max(L))]
            #print('---',L.index(max(L)))
            sigma = self.sigmas[idx, epoch]
        return th.exp(-d ** 2 / (sigma ** 2))

    def kernel_loss(self, k_y, k_l):

        beta = 1.0

        L = th.norm(k_l)
        Y = th.norm(k_y) ** beta
        #X = th.norm(k_x) ** (1-beta)

        LY = th.trace(th.matmul(k_l, k_y))**beta
        #LX = th.trace(th.matmul(k_l, k_x))**(1-beta)

        #return 2*th.log2((LY*LX)/(L*Y*X))
        return 2*th.log2((LY)/(L*Y))

    def cal_mi(self, x,k_x,k_y , model,layer, current_iteration):
      
      data=[k_y]
  
      val=x
  
      k_layer=self.kernel_mat(val.reshape(data[0].size(0), -1),
                                          k_y, epoch=current_iteration,
                                          idx=layer)
      k_list=[]
      k_list.append(k_x)
      k_list.append(k_layer)
      k_list.append(k_y)

      e_list = [self.entropy(i) for i in k_list]

      j_XT = self.entropy(k_list[0], k_list[1])
      j_TY = self.entropy(k_list[1], k_list[2])

      self.MI[current_iteration, layer,0] = e_list[0]+e_list[1]-j_XT
      self.MI[current_iteration, layer,1] = e_list[2]+e_list[1]-j_TY
  
      return


    def weight_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            if self.a_type == 'relu':
                init.kaiming_normal_(m.weight.data, nonlinearity=self.a_type)
                init.constant_(m.bias.data, 0)
            elif self.a_type == 'leaky_relu':
                init.kaiming_normal_(m.weight.data, nonlinearity=self.a_type)
                init.constant_(m.bias.data, 0)
            elif self.a_type == 'tanh':
                g = init.calculate_gain(self.a_type)
                init.xavier_uniform_(m.weight.data, gain=g)
                init.constant_(m.bias.data, 0)
            elif self.a_type == 'sigmoid':
                g = init.calculate_gain(self.a_type)
                init.xavier_uniform_(m.weight.data, gain=g)
                init.constant_(m.bias.data, 0)
            else:
                raise
                return NotImplemented



    def one_hot(self, y, gpu):

        try:
            y = th.from_numpy(y)
        except TypeError:
            None

        y_1d = y
        if gpu:
            y_hot = th.zeros((y.size(0), th.max(y).int()+1)).cuda()
        else:
            y_hot = th.zeros((y.size(0), th.max(y).int()+1))

        for i in range(y.size(0)):
            y_hot[i, y_1d[i].int()] = 1

        return y_hot


    def intialize_layer_name_num(self,model):

        layer_number=0
        for layer_name, layer_module in model.named_modules():
           #if(isinstance(layer_module, th.nn.Conv2d) or  isinstance(layer_module,th.nn.Linear)): -----for both conv and fc layers
           if(isinstance(layer_module, th.nn.Conv2d)):
              self.layer_name_num[layer_number]=layer_name
              layer_number=layer_number+1
        #print('layer name vs number:',self.layer_name_num)
        return


    def filters_in_each_layer(self,model):

        layer_number=0 
        first_conv=-1
        for layer_name, layer_module in model.named_modules():
           #if(isinstance(layer_module, th.nn.Conv2d) or  isinstance(layer_module,th.nn.Linear)):
           if(isinstance(layer_module, th.nn.Conv2d) ):

                  self.pruned_filters[layer_number]=[]
                  self.remaining_filters[layer_number]=[]

                  for i in range(layer_module.weight.size()[0]):
                     self.remaining_filters[layer_number].append(i)
                  layer_number=layer_number+1

        return 

    def remaining_filters_per_epoch(self,model=None,initial=None):
        #print(model,initial)
        if(initial == True):
           layer_number=0 
           for layer_name, layer_module in model.named_modules():
              #if(isinstance(layer_module, th.nn.Conv2d) or  isinstance(layer_module,th.nn.Linear)):
              if(isinstance(layer_module, th.nn.Conv2d)):
                 self.remaining_filters_each_epoch.append([])
                 print(layer_name)
        for i in range(len(self.layer_name_num)):
            self.remaining_filters_each_epoch[i].append(len(self.remaining_filters[i]))
        print('remaining',self.remaining_filters_each_epoch)
        return

    def best_tetr_acc(self,prunes):

      print("prunes vaues id ",prunes)
      tr_acc=self.train_accuracy[prunes:]
      te_acc=self.test_accuracy[prunes:]
      best_te_acc=max(te_acc)
      indices = [i for i, x in enumerate(te_acc) if x == best_te_acc]
      temp_tr_acc=[]
      for i in indices:
         temp_tr_acc.append(tr_acc[i])
      best_tr_acc=max(temp_tr_acc)
      
      del self.test_accuracy[prunes:]
      del self.train_accuracy[prunes:]
      self.test_accuracy.append(best_te_acc)
      self.train_accuracy.append(best_tr_acc)
      return best_te_acc,best_tr_acc

    def compute_bounds(self,outputs,layer):
      #print('layer..',layer,'batch -num=',self.batch_num)

      k_y=self.k_y

      k_list=[]          
      sigma_calculated= self.sigmas[layer,-1]
      #sigma_calculated=self.sigmas_fixed[layer]
      for i in range(outputs.size()[1]): #----i is each filter 

            if( i in self.remaining_filters[layer]):

              #------------below code works only for conv need to change for FC layers

              layer_filter_data=outputs[:,i,:,:]
              k_list.append(self.kernel_mat(layer_filter_data.reshape(outputs.size(0), -1),
              [], sigma=sigma_calculated))
      '''-------------calculating entropies and joint for each filter layer ----------------'''
      k_list.append(k_y)
      e_list = [self.entropy(i) for i in k_list]
      #j_XT = [self.entropy(k_list[0], k_i) for k_i in k_list[1:-1]]
      j_TY = [self.entropy(k_i, k_list[-1]) for k_i in k_list[ :-1]]
      #print('j_ty',len(j_TY))

      for idx_mi, val_mi in enumerate(e_list[ :-1]):

            #self.sub_MI_X[layer][batch_num,idx_mi]=(e_list[0]+val_mi-j_XT[idx_mi])
            #print(idx_mi)
            self.sub_MI_Y[layer][self.batch_num,idx_mi]=(e_list[-1]+val_mi-j_TY[idx_mi])


    def create_folders(self,total_convs):

      main_dir=strftime("%b%d_%H:%M%p", localtime() )+"lenet/"
  
      # Parent Directory path 
      #parent_dir = "/home/sarvani/Documents/code/"+main_dir
      #parent_dir= "/content/drive/MyDrive/information_theory/Results/" + main_dir
      parent_dir= "/home/kishank/Documents/IPPLANE_LENET/Results/" + main_dir
 
      for i in range(total_convs):
        path1=os.path.join(parent_dir, "distributions/conv"+str(i+1))
        os.makedirs(path1)
        path1=os.path.join(parent_dir, "mi_values/conv"+str(i+1))
        os.makedirs(path1)

      return parent_dir

    def get_writerow(self,k):

      s='wr.writerow(['

      for i in range(k):

          s=s+'d['+str(i)+']'

          if(i<k-1):
             s=s+','
          else:
             s=s+'])'

      return s
