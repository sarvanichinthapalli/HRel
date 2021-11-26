import torch.nn.utils.prune as prune
import torch as th
import torch.nn as nn 

filters_selected=[]
no_of_dimensions=-1
prune_percentage=[0.04]+[0.12]
pruned=[]

class PruningMethod(prune.BasePruningMethod):
    """Prune every other entry in a tensor
    """
    PRUNING_TYPE = 'structured'
    dim=0
    def compute_mask(self, t, default_mask):
        global filters_selected
        global no_of_dimensions
        mask = default_mask.clone()
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

    def prune_lenet(self,module,layer_bounds,layer_num,prune_limit):

      global filters_selected
      global no_of_dimensions
      theshold=-0.02


      indices=self.get_indices_topk(layer_bounds,layer_num,prune_limit)  

      for i in range(len(layer_bounds)):
        if(i not in indices):
          filters_selected.append(1)
        else:
          filters_selected.append(0)
          self.pruned_filters[layer_num].append(self.remaining_filters[layer_num][i])

      '''for i,val in enumerate(layer_bounds):
        #print(self.remaining_filters["conv2"][i])
        if val>theshold:
          filters_selected.append(1)
        else:
          filters_selected.append(0)
          self.pruned_filters[layer_name].append(self.remaining_filters[layer_name][i])'''

      self.remaining_filters[layer_num]= sorted(list(set(self.remaining_filters[layer_num])
      -set(self.pruned_filters[layer_num])) )   

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

      #---------------remove filters from selected list to use it in next prune--------------

      filters_selected=[]
      no_of_dimensions=-1

      return module


    def get_indices_top10(self,layer_bounds):
      indices=int(len(layer_bounds)*0.10)+1
      k=sorted(range(len(layer_bounds)), key=lambda i: layer_bounds[i])[:indices]
      return k
    def get_indices_top5(self,layer_bounds):
      indices=int(len(layer_bounds)*0.05)+1
      k=sorted(range(len(layer_bounds)), key=lambda i: layer_bounds[i])[:indices]
      return k
    def get_indices_topk(self,layer_bounds,i,prune_limit):
      global prune_percentage
      indices=int(len(layer_bounds)*prune_percentage[i])+1

      p=len(layer_bounds)
      if (p-indices)<prune_limit:
         remaining=p-prune_limit
         indices=remaining
      k=sorted(range(len(layer_bounds)), key=lambda j: layer_bounds[j])[:indices]
      return k
