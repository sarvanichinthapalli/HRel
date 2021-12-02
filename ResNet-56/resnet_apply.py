import torch.nn.utils.prune as prune
import torch as th
import torch.nn as nn 

filters_selected=[]

no_of_dimensions=-1
prune_value=[1,2,4]

class PruningMethod(prune.BasePruningMethod):
    """Prune every other entry in a tensor
    """
    def __init__(self):
        pass

    def __call__(self, module, inputs):
        r"""Multiplies the mask (stored in ``module[name + '_mask']``)
        into the original tensor (stored in ``module[name + '_orig']``)
        and stores the result into ``module[name]`` by using
        :meth:`apply_mask`.
        Args:
            module (nn.Module): module containing the tensor to prune
            inputs: not used.
        """
        setattr(module, self._tensor_name, self.apply_mask(module))
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
      global prune_value
      i=int(layer_num/18)
      indices_num=prune_value[i]

      p=len(layer_bounds)
      if (p-indices_num)< prune_limit:
         indices_num=p-prune_limit

      indices=self.get_indices_topk(layer_bounds,layer_num,indices_num)

      
      for i in range(len(layer_bounds)):
        if(i not in indices):
          filters_selected.append(1)
        else:
          filters_selected.append(0)
          self.pruned_filters[layer_num].append(self.remaining_filters[layer_num][i])          


      self.remaining_filters[layer_num]= sorted(list(set(self.remaining_filters[layer_num])-set(self.pruned_filters[layer_num])) )   


      #----------------pruning of conv layer weight and bias----------
      #print(module)
      if(isinstance(module, th.nn.Conv2d)):
        no_of_dimensions=4
        
        PruningMethod.apply(module,"weight")
        #bias_mask = (th.sum(module.weight_mask, axis=(1, 2, 3)) != 0).to(th.float32)

      #----------------pruning of FC layer weight and bias----------

      else:
        no_of_dimensions=2
        PruningMethod.apply(module,"weight")
        #bias_mask = (torch.sum(module.weight_mask, axis=(1)) != 0).to(torch.float32)
      

      filters_selected=[]
      no_of_dimensions=-1

      return module

    def get_indices_topk(self,layer_bounds,layer_num,indices):
 
      k=sorted(range(len(layer_bounds)), key=lambda j: layer_bounds[j])[:indices]
      #print('indidces',k)
      return k
