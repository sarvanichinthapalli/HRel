import torch.nn.utils.prune as prune
import torch as th
import torch.nn as nn 

filters_selected=[]
#filters_remaining=[]
no_of_dimensions=-1
prune_percentage=[0.08]*3*2+[0.08]*4*2+[0.08]*6*2+[0.09]*3*2
#prune_value=[1,2,4]

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

    def prune_filters(self,model,module,layer_bounds,layer_num):

      global filters_selected
      global no_of_dimensions
      theshold=-0.02
      

      indices=self.get_indices_topk(layer_bounds,layer_num)  
      
      #print('------remanig filters',self.remaining_filters[layer_num])
      #print('pruned filters',self.pruned_filters[layer_num])

      for i in range(len(layer_bounds)):
        if(i not in indices):
          filters_selected.append(1)
        else:
          filters_selected.append(0)
          model.pruned_filters[layer_num].append(model.remaining_filters[layer_num][i])

      model.remaining_filters[layer_num]= sorted(list(set(model.remaining_filters[layer_num])
      -set(model.pruned_filters[layer_num])) )   

      if(isinstance(module, th.nn.Conv2d)):
        no_of_dimensions=4
        PruningMethod.apply(module,"weight")
        #bias_mask = (th.sum(module.weight_mask, axis=(1, 2, 3)) != 0).to(th.float32)

      #----------------pruning of FC layer weight and bias----------

      else:
        no_of_dimensions=2
        PruningMethod.apply(module,"weight")
        #bias_mask = (torch.sum(module.weight_mask, axis=(1)) != 0).to(torch.float32)
        
     
      #prune.custom_from_mask(module, 'bias', bias_mask)

      #---------------remove filters from selected list to use it in next prune--------------

      filters_selected=[]
      no_of_dimensions=-1

      return module


      
    def get_indices_topk(self,layer_bounds,i,prune_limit=0):

      global prune_percentage
      indices=int(len(layer_bounds)*prune_percentage[i]) #1

      k=sorted(range(len(layer_bounds)), key=lambda j: layer_bounds[j])[:indices]

      print('indidces',k, 'len ',len(layer_bounds))
      return k  

    def resume_prune_filters(self,model,module,pruned_filters,layer_num):
      global filters_selected
      global no_of_dimensions
      for i in range(module.weight.size(0)):

        if(i not in pruned_filters):
          filters_selected.append(1)
        else:
          filters_selected.append(0)

          self.pruned_filters[layer_num].append(self.remaining_filters[layer_num][i])


      self.remaining_filters[layer_num]= sorted(list(set(self.remaining_filters[layer_num])
      -set(self.pruned_filters[layer_num])) )   
      print('remanig filtersof layer ',layer_num,'=',len(self.remaining_filters[layer_num]))
      #print('------pruned filters',self.pruned_filters[layer_num],'\n\n')

      
      if(isinstance(module, th.nn.Conv2d)):
        no_of_dimensions=4
        PruningMethod.apply(module,"weight")
        #bias_mask = (th.sum(module.weight_mask, axis=(1, 2, 3)) != 0).to(th.float32)

      else:
        no_of_dimensions=2
        PruningMethod.apply(module,"weight")
        #bias_mask = (torch.sum(module.weight_mask, axis=(1)) != 0).to(torch.float32)
        
     
      #prune.custom_from_mask(module, 'bias', bias_mask)

      #---------------remove filters from selected list to use it in next prune--------------
      
      filters_selected=[]
      no_of_dimensions=-1
      print('pruned filters of layer ',layer_num,'=',len(self.pruned_filters[layer_num]))
      return

