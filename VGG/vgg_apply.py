import torch.nn.utils.prune as prune
import torch as th
import torch.nn as nn 

filters_selected=[]
#filters_remaining=[]
no_of_dimensions=-1
prune_percentage=[0.02]*2+[0.04]*2+[0.05]*3+[0.1]*6


class PruningMethod(prune.BasePruningMethod):
    PRUNING_TYPE = 'structured'
    dim=0
    def compute_mask(self, t, default_mask):
        global filters_selected
        global no_of_dimensions
        mask = default_mask.clone()
        #print("the mask size is ",mask.size())
        #print(self.pruned_filters["conv2"])
        if(no_of_dimensions==4):
          for i,val in enumerate(filters_selected):
            if(val==0):
              mask[i,:,:,:]=0
        if(no_of_dimensions==2):
          for i,val in enumerate(filters_selected):
            if(val==0):
              mask[i,:]=0
        if(no_of_dimensions==1):
          for i,val in enumerate(filters_selected):
            if(val==0):
              mask[i]=0  
        return mask

    def prune_lenet(self,module,layer_bounds,layer_num):

      global filters_selected
      global no_of_dimensions
      theshold=-0.02
      

      #if(layer_name=="conv2"):
      #  indices=self.get_indices_top10(layer_bounds)
      #else:
      #  indices=self.get_indices_top5(layer_bounds)

      indices=self.get_indices_topk(layer_bounds,layer_num)  

      for i in range(len(layer_bounds)):
        if(i not in indices):
          filters_selected.append(1)
        else:
          filters_selected.append(0)
          self.pruned_filters[layer_num].append(self.remaining_filters[layer_num][i])

      self.remaining_filters[layer_num]= sorted(list(set(self.remaining_filters[layer_num])
      -set(self.pruned_filters[layer_num])) )   
      #print('remaning filters',self.remaining_filters[layer_num])
      #print('------pruned filters',self.pruned_filters[layer_num],'\n\n')

      #----------------pruning of conv layer weight and bias----------
      #print(module)
      if(isinstance(module, th.nn.Conv2d)):
        no_of_dimensions=4
        PruningMethod.apply(module,"weight")
        bias_mask = (th.sum(module.weight_mask, axis=(1, 2, 3)) != 0).to(th.float32)

      #----------------pruning of FC layer weight and bias----------

      else:
        no_of_dimensions=2
        PruningMethod.apply(module,"weight")
        bias_mask = (torch.sum(module.weight_mask, axis=(1)) != 0).to(torch.float32)
        
     
      prune.custom_from_mask(module, 'bias', bias_mask)


      filters_selected=[]
      no_of_dimensions=-1

      return module

    def get_indices_topk(self,layer_bounds,i):
      global prune_percentage
      indices=int(len(layer_bounds)*prune_percentage[i])+1
      k=sorted(range(len(layer_bounds)), key=lambda j: layer_bounds[j])[:indices]
      #print('indidces',k)
      return k
