
import torch as th
import torch.nn as nn
import torch.nn.init as init
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import csv 
from time import localtime, strftime
import os 
from itertools import zip_longest
seed = 1787
#random.seed(seed)
#import os
#os.environ['PYTHONHASHSEED'] = str(seed)

#th.manual_seed(seed)
#th.cuda.manual_seed(seed)
#th.cuda.manual_seed_all(seed)
#th.backends.cudnn.deterministic = True


class Network():


    def predict(self, x, y, model, gpu):

        model.eval()
        output = model(x)
        y_hat = th.argmax(self.softmax(output[0]), 1)
        
        score = th.eq(y_hat, y).sum()

        return score.item()

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

        eigv = th.abs(th.symeig(k, eigenvectors=False)[0])
        temp=eigv.clone()
        eigv_log2= temp.log2().cuda()

        if((eigv==c).any()):

          zero_indices=(eigv == 0).nonzero().tolist()
          small=th.tensor([0.0000000099]).cuda()
          small_value=small.detach().clone()
          for i in zero_indices:
            eigv_log2[i]=small_value

        return -(eigv*(eigv_log2)).sum()

    def kernel_mat(self, x, k_y, sigma=None, epoch=None, idx=None):

        d = self.dist_mat(x)
        #print(idx,'-----',d.mean())
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

    def compute_mi(self, x, y, model, gpu, current_iteration):
       
        model.eval()

        data = model(x)
        data.reverse()

        data[-1] = self.softmax(data[-1])
        data.insert(0, x)
        data.append(self.one_hot(y, gpu))

        k_x = self.kernel_mat(data[0], [], sigma=th.tensor(8.0))
        
        k_y = self.kernel_mat(data[-1], [], sigma=th.tensor(0.1))

        k_list = [k_x]
        #k_list=[]
        for idx_l, val in enumerate(data[1:-1]):
   
            k_list.append(self.kernel_mat(val.reshape(data[0].size(0), -1),
                                          k_y, epoch=current_iteration,
                                          idx=idx_l))
        k_list.append(k_y)

        e_list = [self.entropy(i) for i in k_list]

        j_XT = [self.entropy(k_list[0], k_i) for k_i in k_list[1:-1]]

        j_TY = [self.entropy(k_i, k_list[-1]) for k_i in k_list[ :-1]]

        for idx_mi, val_mi in enumerate(e_list[1 :-1]):
            #print(current_iteration,idx_mi)
            self.MI[current_iteration, idx_mi, 0] = e_list[0]+val_mi-j_XT[idx_mi]
            self.MI[current_iteration, idx_mi,1] = e_list[-1]+val_mi-j_TY[idx_mi]


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

    def epoch_accuracy(self,s,e,per_epoch):
        start=s
        end=e
        acc=(sum(self.score[start:end])*100)/per_epoch
        return acc
    


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
        for layer_name, layer_module in model.named_modules():
           #if(isinstance(layer_module, th.nn.Conv2d) or  isinstance(layer_module,th.nn.Linear)):
           if(isinstance(layer_module, th.nn.Conv2d)):

              self.pruned_filters[layer_number]=[]
              self.remaining_filters[layer_number]=[]

              for i in range(layer_module.weight.size()[0]):
                 self.remaining_filters[layer_number].append(i)
              layer_number=layer_number+1
        return 



    def remaining_filters_per_epoch(self,model=None,initial=None):
        if(initial == True):
           layer_number=0 
           for layer_name, layer_module in model.named_modules():
              #if(isinstance(layer_module, th.nn.Conv2d) or  isinstance(layer_module,th.nn.Linear)):
              if(isinstance(layer_module, th.nn.Conv2d)):
                 self.remaining_filters_each_epoch.append([])
                 #print(layer_name)
        for i in range(len(self.layer_name_num)):
            self.remaining_filters_each_epoch[i].append(len(self.remaining_filters[i]))
        print('remaining filters',self.remaining_filters_each_epoch)
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

    def compute_bounds(self,x, y, model, gpu,batch_num, current_iteration,layers_for_pruning,folder_name,decision_count,no_of_batches):

      #-----------get layerwise ouput for given batch of data---------------
      model.eval()
      data = model(x)
      data.reverse()
      data[-1] = self.softmax(data[-1])
      data.insert(0, x)
      data.append(self.one_hot(y, gpu))

      #k_x = self.kernel_mat(data[0], [], [], sigma=th.tensor(8.0))
        
      k_y = self.kernel_mat(data[-1], [], sigma=th.tensor(0.1))
      
      a=[]
      
      #-------------------start computing MI for each layer and each filter-------------

      for layer in range(layers_for_pruning):

          #-------------size of output from the hidden layer----------
          a=data[layer+1].size()    

          #-------------ouput from the hidden layer------------
          layer_data=data[layer+1] 

          #-------no.of REMAINING  filters in given layer--------- 
          filters_num= int(a[1]- len(self.pruned_filters[layer]))

          #print("layer..",str(layer+1),"...filters...",str(filters_num),"..batch..",str(batch_num))
          
          if(batch_num==0):
              #self.sub_MI_X.append(th.zeros((no_of_batches,filters_num)).cuda())
              self.sub_MI_Y.append(th.zeros((no_of_batches,filters_num)).cuda())
          if(decision_count[layer]==0):
              continue
          #k_list = [k_x]
          k_list=[]          
          sigma_calculated= self.sigmas_fixed[layer]
          #-------------calculating kernel matrix for each filter layer ----------------

          for i in range(int(a[1])): #----i is each filter 

            if( i in self.remaining_filters[layer]):

              #------------below code works only for conv need to change for FC layers

              layer_filter_data=layer_data[:,i,:,:]
              k_list.append(self.kernel_mat(layer_filter_data.reshape(data[0].size(0), -1),
              [], sigma=sigma_calculated))

          

          '''-------------calculating entropies and joint for each filter layer ----------------'''
          k_list.append(k_y)
          e_list = [self.entropy(i) for i in k_list]
          #j_XT = [self.entropy(k_list[0], k_i) for k_i in k_list[1:-1]]
          j_TY = [self.entropy(k_i, k_list[-1]) for k_i in k_list[ :-1]]


          '''-------------calculating MI for each filter layer ----------------'''
          for idx_mi, val_mi in enumerate(e_list[ :-1]):

            #self.sub_MI_X[layer][batch_num,idx_mi]=(e_list[0]+val_mi-j_XT[idx_mi])
            self.sub_MI_Y[layer][batch_num,idx_mi]=(e_list[-1]+val_mi-j_TY[idx_mi])

            if(th.isnan(self.sub_MI_Y[layer][batch_num,idx_mi])):
              print(e_list[-1],val_mi,j_TY[idx_mi])

      #------if final batch save all outputs plots and return bounds for all layers over all batches
      if(batch_num+1 == no_of_batches):
                             
            beta=2
            z=[]
            small_z=[]
            
            for i in range(layers_for_pruning):#-------------------2 conv layers

              z.append((th.sum(self.sub_MI_Y[i],dim=0)).div(no_of_batches).tolist())
              small_z.append((th.sum(self.sub_MI_Y[i][:200,:],dim=0)).div(200).tolist())
              #layers_x=[]
              layers_y=[]
              

              z[-1]=[float("{:.7f}".format(i)) for i in z[-1]]
              small_z[-1]=[float("{:.7f}".format(i)) for i in small_z[-1]]
            

            z=[[i for i in lst] for lst in z]

            for i in range(layers_for_pruning):

              plt.hist(z[i], weights=np.ones(len(z[i])) / len(z[i]))
              plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
              plt.savefig(folder_name+"distributions/conv"+
              str(i+1)+"/iter"+str(current_iteration)+"_"+str(len(self.remaining_filters[i]))+".png")
              plt.clf()
              plt.cla()
              plt.close()

              fig, ax = plt.subplots(figsize=(60,10))
              pcm = ax.pcolormesh(self.sub_MI_Y[i].cpu().numpy())
              fig.colorbar(pcm, ax=ax)
              plt.savefig(folder_name+"mi_values/conv"+
              str(i+1)+"/iter"+str(current_iteration)+"_"+str(len(self.remaining_filters[i]))+".png")
              plt.show() #comment this to stop viewing at run time
              plt.close()

            self.sub_MI_X=[]
            self.sub_MI_Y=[]
            return z

      return 0

    def create_folders(self,total_convs):

      main_dir=strftime("%d%b_%H:%M%p", localtime() )+"_vgg_ip_plane2/"
  
      # Parent Directory path 
      #parent_dir = "/home/sarvani/Documents/code/"+main_dir
      #parent_dir= "/content/drive/MyDrive/information_theory/Results/" + main_dir
      parent_dir= "/home/hareesh/Documents/sar/information_theory/Results/" + main_dir
 
      for i in range(13):
        path1=os.path.join(parent_dir, "distributions/conv"+str(i+1))
        os.makedirs(path1)
        path1=os.path.join(parent_dir, "mi_values/conv"+str(i+1))
        os.makedirs(path1)

      return parent_dir

