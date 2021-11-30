
import torch as th
import torch.nn as nn
import torch.nn.init as init
#import numpy as np
import time, datetime
import logging
import csv 
from time import localtime, strftime
import os 



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
        #print(k.size())
        c = th.tensor([0]).to('cuda:0')
        #print(not(th.symeig(k)[0] < c).any())
        try: 
          eigv = th.abs(th.symeig(k, eigenvectors=False)[0]) #1D array = batch-size
        except:
          print('NO eig values')
          return None
          
        #eigv=th.sum(th.sqrt(th.diag(th.matmul(k.t(), k))))
        #print('eigen values=',eigv.size())
        #print(eigv)
        temp=eigv.clone()
        eigv_log2= temp.log2().to('cuda:0')
        if((eigv==c).all()):
           print('loooong')
        if((eigv==c).any()):
          #print('smalll')	          
          zero_indices=(eigv == 0).nonzero().tolist()
          small=th.tensor([0.0000000099]).to('cuda:0')
          small_value=small.detach().clone()
          for i in zero_indices:
            eigv_log2[i]=small_value
          #print(eigv*eigv_log2.sum())

        return -(eigv*(eigv_log2)).sum()

    def kernel_mat(self, model,x, k_y, sigma=None, epoch=None, idx=None):
        #print('kernel mat',x.size())
        d = self.dist_mat(x).to('cuda:0')
        #print(idx,'-----',d.mean())
        c = th.tensor([0]).to('cuda:0')
        if sigma is None:
            if epoch > 10:
                sigma_vals = th.linspace(0.3, 10*d.mean(), 100).to('cuda:0')
                #sigma_vals = th.linspace(0.1*d.mean(), 10*d.mean(), 50).cuda() 
             
            else:
                sigma_vals = th.linspace(0.3, 10*d.mean(), 300).to('cuda:0') 
                #sigma_vals = th.linspace(0.1*d.mean(), 10*d.mean(), 75).cuda()
            L = []
            for sig in sigma_vals:
                k_l = th.exp(-d ** 2 / (sig ** 2)) / d.size(0)
                #k_l=k_l.to('cuda:0')
                L.append(self.kernel_loss(k_y, k_l))

            if epoch == 0:
                model.sigmas[idx, epoch] = sigma_vals[L.index(max(L))]
            else:
                while(model.sigmas[idx, epoch-1]==c):
                   continue
                   #print('idx=',idx,'iteration=',epoch,'new=',model.sigmas[idx, epoch],'old=',model.sigmas[idx, epoch-1])
                   #print('updated=',model.sigmas[idx, epoch-1])
                model.sigmas[idx, epoch] = 0.9*model.sigmas[idx, epoch-1] + 0.1*sigma_vals[L.index(max(L))]

            sigma = model.sigmas[idx, epoch]
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

    def cal_mi(self, x,k_x,k_y , model,layer_name, current_iteration,sigma_update):
      
      #print('input is...',x.s'output is...',y.size(),'layer..',layer,'iter...',current_iteration)
      #print(x.size(),'<---------------->',x.device)
      a=10
      layer=model.layer_name_num[layer_name]
      #print('********************************')
      #print('layer=',layer,'mi_iteration=',current_iteration,x.size(),'device=',x.device,'k_y=',k_y.size())
      data=[k_y]
  
      val=x 
      #print('shape',val.reshape(data[0].size(0), -1).size() )
      k_layer=self.kernel_mat(model,val.reshape(data[0].size(0), -1),k_y, epoch=current_iteration,idx=layer)

      return


    def cal_mi_durga(self, x,k_x,k_y , model,layer_name, current_iteration,sigma_update):
      
      #print('input is...',x.s'output is...',y.size(),'layer..',layer,'iter...',current_iteration)
      #print(x.size(),'<---------------->',x.device)
      a=10
      layer=model.layer_name_num[layer_name]
      #print('********************************')
      #print('layer=',layer,'mi_iteration=',current_iteration,x.size(),'device=',x.device,'k_y=',k_y.size())
      data=[k_y]
  
      val=x 
      #print('shape',val.reshape(data[0].size(0), -1).size() ) 
      if(sigma_update==True):
          k_layer=self.kernel_mat(model,val.reshape(data[0].size(0), -1),k_y, epoch=current_iteration,idx=layer)
          #print(data[0].size(0))
      else:
          sigma_calculated= model.sigmas_fixed[layer]
          #print('sigma_calculated for layer ',layer)
          k_layer=self.kernel_mat(model,val.reshape(data[0].size(0), -1),[], sigma=sigma_calculated)
          
      k_list=[]
      k_list.append(k_x)
      k_list.append(k_layer)
      k_list.append(k_y)

      e_list=[]
      for i in k_list:
        temp=self.entropy(i)
        if(temp==None):
          model.ignore_iterations.append(current_iteration)
          model.MI[current_iteration, :,0] = 0
          model.MI[current_iteration, :,1] = 0
          l=0
          for handle in model.durga_handles:
                if(l!=layer): #except ffor this layer remove handles for all layers. handle is only executing this
                  handle.remove()
                l+=1
          return

        e_list.append(temp)

      #e_list = [self.entropy(i) ]
      

      j_XT = self.entropy(k_list[0], k_list[1])
      j_TY = self.entropy(k_list[1], k_list[2])

      layer=model.MI_layer_name_num[layer_name]

      model.MI[current_iteration, layer,0] = e_list[0]+e_list[1]-j_XT
      model.MI[current_iteration, layer,1] = e_list[2]+e_list[1]-j_TY
  
      return

    
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

    def compute_bounds(self,model,outputs,layer_name,num_gpu,gpu_id,logger):
      
      layer=model.layer_name_num[layer_name]
      if(num_gpu==2):
         if(gpu_id=="0"):
            batch_number=model.batch_number* num_gpu
            k_y=model.k_y[0]
         elif(gpu_id=="1"):
            batch_number= (model.batch_number* num_gpu)+1
            k_y=model.k_y[1]
         else:
            print("GPU not present in the list")
      else:
          batch_number= model.batch_number    
          k_y=model.k_y

      k_list=[]          
      sigma_calculated=model.sigmas_fixed[layer]

      for i in range(outputs.size()[1]): #----i is each filter 
          if( i in model.remaining_filters[layer]):
              layer_filter_data=outputs[:,i,:,:]
              k_list.append(model.module.kernel_mat(model,layer_filter_data.reshape(outputs.size(0), -1),
              [], sigma=sigma_calculated))
      
      k_list.append(k_y)
      e_list = [self.entropy(i) for i in k_list]

      #j_XT = [self.entropy(k_list[0], k_i) for k_i in k_list[1:-1]]
      j_TY = [self.entropy(k_i, k_list[-1]) for k_i in k_list[ :-1]]


      for idx_mi, val_mi in enumerate(e_list[ :-1]):
        if(val_mi==None or j_TY[idx_mi]==None):
          model.sub_MI_Y[layer][batch_number,idx_mi]=0
        else:
            
            model.sub_MI_Y[layer][batch_number,idx_mi]=(e_list[-1]+val_mi-j_TY[idx_mi])
            #print(model.sub_MI_Y[layer].size(), batch_number,idx_mi)
      #logger.info('layer '+str(layer)+'completed on '+str(gpu_id))



    def intialize_layer_name_num(self,model):

      layer_number=0
      for layer_name, layer_module in model.named_modules():
         #if(isinstance(layer_module, th.nn.Conv2d) or  isinstance(layer_module,th.nn.Linear)): -----for both conv and fc layers
         if(isinstance(layer_module, th.nn.Conv2d) and layer_name != "module.conv1" and layer_name != "conv1" and layer_name.find('conv3')==-1 and layer_name.find('downsample')==-1):
             #print(layer_name)
             model.layer_name_num[layer_name]=layer_number
             layer_number=layer_number+1
         #print('layer name vs number:',self.layer_name_num)
      return


    def intialize_MI_layer_name_num(self,model):

      layer_number=0
      for layer_name, layer_module in model.named_modules():
         #if(isinstance(layer_module, th.nn.Conv2d) or  isinstance(layer_module,th.nn.Linear)): -----for both conv and fc layers
         if(isinstance(layer_module, th.nn.Conv2d) and layer_name != "module.conv1" and layer_name != "conv1" and layer_name.find('conv1')!=-1 and layer_name.find('downsample')==-1):
             #print(layer_name)
             model.MI_layer_name_num[layer_name]=layer_number
             layer_number=layer_number+1
         #print('layer name vs number:',self.layer_name_num)
      return


    def filters_in_each_layer(self,model):

        layer_number=0 
        first_conv=-1
        for layer_name, layer_module in model.named_modules():

           if(isinstance(layer_module, th.nn.Conv2d) and layer_name != "module.conv1" and layer_name != "conv1" and layer_name.find('conv3')==-1 and layer_name.find('downsample')==-1):

                  model.pruned_filters[layer_number]=[]
                  model.remaining_filters[layer_number]=[]

                  for i in range(layer_module.weight.size()[0]):
                     model.remaining_filters[layer_number].append(i)
                  layer_number=layer_number+1
        return 



    def remaining_filters_per_epoch(self,model=None,initial=None):
        if(initial == True):
           layer_number=0 
           for layer_name, layer_module in model.named_modules():
              if(isinstance(layer_module, th.nn.Conv2d) and layer_name != "module.conv1" and layer_name != "conv1" and layer_name.find('conv3')==-1 and layer_name.find('downsample')==-1 ):
                 model.remaining_filters_each_epoch.append([])
                 #print(layer_name)
        for i in range(len(model.layer_name_num)):
            #print(i,len(model.remaining_filters[i]))
            model.remaining_filters_each_epoch[i].append(len(model.remaining_filters[i]))
        print('remaining',model.remaining_filters_each_epoch)
        return


    def create_folders(self,total_convs):

      main_dir=strftime("/Results/%d%b_%H:%M%p", localtime() )+"_resnet50/"
      #main_dir=strftime("/Results/short_resnet50/"
  
      # Parent Directory path 
      #parent_dir = "/home/sarvani/Documents/code/"+main_dir
      #parent_dir= "/content/drive/MyDrive/information_theory/Results/" + main_dir
      import os

      par_dir=os.path.dirname(os.path.realpath(__file__))
      parent_dir=par_dir+main_dir
      #parent_dir= "/home/hareesh/Documents/sar/resnet_50/Zeroing/Results/" + main_dir
 
      for i in range(total_convs):
        path1=os.path.join(parent_dir, "distributions/conv"+str(i+1))
        os.makedirs(path1)
        path1=os.path.join(parent_dir, "mi_values/conv"+str(i+1))
        os.makedirs(path1)
      path2=os.path.join(parent_dir, "layer_file_info")
      os.makedirs(path2)
      return parent_dir

    def get_writerow(self,k):

      s='wr.writerow(['

      for i in range(k):

          s=s+'d['+str(i)+']'

          if(i<k-1):
             s=s+','
          else:
             s=s+'])'
      #print(s)
      return s
