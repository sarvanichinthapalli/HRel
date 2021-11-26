''' 1.Traning: whole train data
    2.Testing: Whole test data batch wise'''
import random
import torch as th
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
#from CNN_Hero import CNN_Hero
from updated_LeNet import LeNet
import csv
from itertools import zip_longest
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from matplotlib.ticker import PercentFormatter
from torch.optim.lr_scheduler import MultiStepLR

seed =1787
random.seed(seed)
import os
os.environ['PYTHONHASHSEED'] = str(seed)

th.manual_seed(seed)
th.cuda.manual_seed(seed)
th.cuda.manual_seed_all(seed)
th.backends.cudnn.deterministic = True
#os.environ["CUDA_VISIBLE_DEVICES"]="0"


th.cuda.set_device(0)

trainloader = th.utils.data.DataLoader(datasets.MNIST('../mnist_data', 
                                                          download=True, 
                                                          train=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                              #transforms.Normalize((0.5,), (0.5,)) # normalize inputs
                                                          ])), 
                                           batch_size=100, 
                                           shuffle=True,num_workers=2)

# download and transform test dataset
testloader = th.utils.data.DataLoader(datasets.MNIST('../mnist_data', 
                                                          download=True, 
                                                          train=False,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                              #transforms.Normalize((0.5,), (0.5,)) # normalize inputs
                                                          ])), 
                                           batch_size=100, 
                                           shuffle=True,num_workers=2)

N = 1

batch_size_tr = 100
batch_size_te = 100

epochs = 40
new_epochs=40
mi_epoch=0
mi_calc_epochs=10

total_layers=5
total_convs=2
total_blocks=total_convs

prune_limits=[4,5]
#prune_limits=[17,45]

decision_count=th.ones((total_convs))

tr_size = 60000
te_size=10000


activation = 'relu'

#tr_size = 300
#te_size=300
short_train=False
n_iterations_per_epoch= (tr_size // batch_size_tr)
n_iterations = (tr_size // batch_size_tr)*(epochs - mi_epoch)

print('n_iterations.....######################################',n_iterations)


model=LeNet( n_iterations).cuda()

gpu=th.cuda.is_available()

with th.no_grad():

  folder_name=model.create_folders(total_convs)
  #---------intialize layer numbers and names dictionary------
  model.intialize_layer_name_num(model)

  #---------intialize pruned and remaning filters-----------
  model.filters_in_each_layer(model)
  
  #--------calculate remaining filters in each epoch--------
  model.remaining_filters_per_epoch(model=model,initial=True)


#optimizer = th.optim.SGD(model.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
optimizer = th.optim.SGD(model.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
#scheduler = MultiStepLR(optimizer, milestones=[80,150,240], gamma=0.1)
scheduler = MultiStepLR(optimizer, milestones=[20,30], gamma=0.1)

print(model.sigmas.shape)
criterion = nn.CrossEntropyLoss()
for n in range(N):

    current_iteration = 0
    mi_iteration=0

    for epoch in range(epochs):

      start=current_iteration


      train_acc=[]

      for batch_num, (inputs, targets) in enumerate(trainloader):

        if(batch_num==3 and (short_train == True)):
                break
        inputs = inputs.cuda()
        targets = targets.cuda()

        model.train()

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        with th.no_grad():

          y_hat = th.argmax(output, 1)
          score = th.eq(y_hat, targets).sum()

          train_acc.append(score.item())

          #---------------------------hooks code start from here-----------
          if(epoch>=mi_epoch):
            #print('mi=',mi_iteration,'cur=',current_iteration)
            if(mi_iteration%100==0):
               print("you entered epoch",epoch,'iterations ',current_iteration%600)
            #print("you entered epoch",epoch,'iterations ',current_iteration)
            with th.no_grad():

              y=model.one_hot(targets, True)
              k_y = model.kernel_mat(y, [], sigma=th.tensor(0.1))
              k_x = model.kernel_mat(inputs, [], sigma=th.tensor(8.0))
              def forward_hook(layer_name):
                      def hook(module, input, output):
                          #print(layer_name,targets.size())
                          model.cal_mi(output,k_x, k_y, model,layer_name, mi_iteration)
                      return hook
              
              handles=[]
              layer=0

              for name, layer_module in model.named_modules():

                  if(isinstance(layer_module, th.nn.Conv2d) or isinstance(layer_module,th.nn.Linear)):

                    #print(name,layer_module)
                    handles.append(layer_module.register_forward_hook(forward_hook(layer)))
                    layer=layer+1

              model.eval()
              model(inputs)

              layer=0
              for handle in handles:
                handle.remove()
              mi_iteration+=1

          current_iteration += 1

          #---------------------------hooks end here------------------------------

      
      with th.no_grad():

        model.train_accuracy.append((sum(train_acc)*100)/tr_size)
            
        test_acc=[]
        model.eval()
        for batch_nums, (inputs2, targets2) in enumerate(testloader):
            if(batch_nums==3 and (short_train == True)):
                break

            inputs2, targets2 = inputs2.cuda(), targets2.cuda()
             
            output=model(inputs2)
            y_hat = th.argmax(output, 1)
            score = th.eq(y_hat, targets2).sum()

            test_acc.append(score.item())


        model.test_accuracy.append((sum(test_acc)*100)/te_size)        

      end=current_iteration

      print('\n---------------Epoch number: {}'.format(epoch),
              '---Train accuracy: {}'.format(model.train_accuracy[-1]),
              '----Test accuracy: {}'.format(model.test_accuracy[-1]),'--------------')

      scheduler.step()
      print(optimizer.param_groups[0]['lr'])

ended_epoch=epoch
ended_iteration=end

prunes=0

final_train_acc= model.train_accuracy[-1]
final_test_acc= model.test_accuracy[-1]
del model.test_accuracy[prunes:]
del model.train_accuracy[prunes:]
model.test_accuracy.append(final_test_acc)
model.train_accuracy.append(final_train_acc)


#----------------writing data---------------------
    
d=[]
for i in range(total_convs):
      d.append(model.remaining_filters_each_epoch[i][-1])

d.append(model.train_accuracy[-1])
d.append(model.test_accuracy[-1])

with open(folder_name+'vggPrune.csv', 'a', newline='') as myfile:
          wr = csv.writer(myfile)

          command=model.get_writerow(total_convs+2)
          eval(command)

myfile.close()

a=[]

for layer_name, layer_module in model.named_modules():

  if(isinstance(layer_module, th.nn.Conv2d)):
    a.append(layer_module)

model.sigmas_fixed[0]=model.sigmas[0,-1]
model.sigmas_fixed[1]=model.sigmas[1,-1]


decision=True
#----------------------retrain the model--------------------
while(decision==True):
    #----------------------prune the model--------------------


    with th.no_grad():

       #-----------------------hook creation start -----------------

      def forward_hook(layer_name):
                      def hook(module, input, output):
                          #print(layer_name,output.size())
                          model.compute_bounds(output,layer_name)
                      return hook
              
      handles=[]
      layer=0
      for name, layer_module in model.named_modules():

          if(isinstance(layer_module, th.nn.Conv2d)):

             handles.append(layer_module.register_forward_hook(forward_hook(layer)))
             layer=layer+1

      #-----------------------hook creation end -----------------
      
      no_of_batches=(tr_size // batch_size_tr)
      z=[]
      small_z=[]
      model.eval()

      for batch_num, (inputs, targets) in enumerate(trainloader):

        if(batch_num==3 and (short_train == True)):
          break
        inputs = inputs.cuda()
        targets = targets.cuda()
        
        if(batch_num==0):  #------------------Intilaize sub_mi_y if first batch------------------

            layer=0
            for name, layer_module in model.named_modules():

                if(isinstance(layer_module, th.nn.Conv2d)):

                    filters_num=model.remaining_filters[layer]
                    model.sub_MI_Y.append(th.zeros((no_of_batches,len(filters_num))).cuda())
                    layer=layer+1

        y=model.one_hot(targets, gpu)
        model.k_y=model.kernel_mat(y, [], sigma=th.tensor(0.1)) #-------------update kernel matrix for y
        model.batch_num=batch_num                              #--------------update batch number
        
        
        #----------remove hooks for layers that reached maximum to avoid computing bounds-----------------
        layer=0

        for name, layer_module in enumerate(model.named_modules()):
          if(isinstance(layer_module, th.nn.Conv2d)):
              if(len(model.remaining_filters[layer]) <= prune_limits[layer] ):
                  handles[layer].remove()
              layer=layer+1

                

        model(inputs) #---------hook called here and bounds calculated 
 
        if(batch_num+1 == no_of_batches): #--------------create plots for data--------------

          beta=2

            
          for i in range(total_convs):

              l1norm=[]

              for layer_name, layer_module in model.named_modules():
                 
                 if(layer_name == model.layer_name_num[i]):
                    filter_weight=layer_module.weight.clone()
                    for k in range(filter_weight.size()[0]):
                      if(k in model.remaining_filters[i]):
                        l1norm.append(float("{:.6f}".format(th.norm(filter_weight[k,:,:,:]).item())))

              z.append((th.sum(model.sub_MI_Y[i],dim=0)).div(no_of_batches).tolist())
              #z.append(l1norm)
              z[-1]=[float("{:.7f}".format(i)) for i in z[-1]]

          z=[[i for i in lst] for lst in z]

          for i in range(total_convs):

            plt.hist(z[i], weights=np.ones(len(z[i])) / len(z[i]))
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
            plt.savefig(folder_name+"distributions/conv"+
            str(i+1)+"/iter"+str(current_iteration)+"_"+str(len(model.remaining_filters[i]))+".png")
            plt.clf()
            plt.cla()
            plt.close()

            fig, ax = plt.subplots(figsize=(60,10))
            pcm = ax.pcolormesh(model.sub_MI_Y[i].cpu().numpy())
            fig.colorbar(pcm, ax=ax)
            plt.savefig(folder_name+"mi_values/conv"+
            str(i+1)+"/iter"+str(current_iteration)+"_"+str(len(model.remaining_filters[i]))+".png")
            #plt.show()
            plt.close()

          model.sub_MI_X=[]
          model.sub_MI_Y=[]

      layer_bounds1=z

      
      #-------------------pruning each conv layer and removing hooks---------------
      for i in range(len(layer_bounds1)-1,-1,-1):
          #print('the layer is...',i)
          handles[i].remove()
          if(len(model.remaining_filters[i]) > prune_limits[i] ):	
            model.prune_lenet(a[i],layer_bounds1[i],i,prune_limits[i] )
          else:
            print(model.layer_name_num[i],"has reached",prune_limits[i] )
            if(decision_count[i]!=0):
                decision_count[i]=0

      if(th.sum(decision_count)==0):
          decision=False  
 
      prunes+=1
      model.remaining_filters_per_epoch(model=model)


    ended_accuracy=model.test_accuracy[-1]


    #-----------------------------m3--------------
    new_iterations=((tr_size // batch_size_tr)*new_epochs)
    #----------------------------m4----------------
    #new_iterations=((te_size // batch_size_te)*new_epochs)

    print('new-model starts....for ',new_iterations,' iterations ,ended epoch=',ended_epoch)
    '''layer=0
    for name, layer_module in model.named_modules():
          if(isinstance(layer_module, th.nn.Conv2d) and name != "conv1"):
              print('layer',layer,'..',layer_module._forward_hooks)
              layer=layer+1'''

    with th.no_grad():
      new_mi_iterations=((tr_size // batch_size_tr)*mi_calc_epochs)
      new_MI = th.zeros((new_mi_iterations,total_layers, 2)).cuda() #-----------change to 16 for vgg
      new_sigmas = th.zeros((total_layers, new_mi_iterations)).cuda() #---------change to 16 for VGG
      model.MI=th.cat((model.MI,new_MI))
      model.sigmas= th.cat((model.sigmas,new_sigmas),dim=1)
      print(model.MI.shape,model.sigmas.shape)
    #optimizer = th.optim.SGD(model.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
    optimizer.param_groups[0]['lr']=0.1
    scheduler = MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)
    
    for epoch in range(new_epochs):

          start=current_iteration
          train_acc=[]
          test_acc=[]

          for batch_num, (inputs, targets) in enumerate(trainloader):

            if(batch_num==3 and (short_train == True)):
               break

            inputs = inputs.cuda()
            targets = targets.cuda()

            model.train()

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            with th.no_grad():

              y_hat = th.argmax(output, 1)
              score = th.eq(y_hat, targets).sum()

              train_acc.append(score.item())

              if(epoch>=(new_epochs-mi_calc_epochs)):#30) #----change new_MI ,new_sigmas also
                      y=model.one_hot(targets, True)
                      k_y = model.kernel_mat(y, [], sigma=th.tensor(0.1))
                      k_x = model.kernel_mat(inputs, [], sigma=th.tensor(8.0))
                      def forward_hook(layer_name):
		                      def hook(module, input, output):
		                          #print(layer_name,targets.size())
		                          model.cal_mi(output,k_x, k_y, model,layer_name, mi_iteration)
		                      return hook
		              
                      handles=[]
                      layer=0

                      for name, layer_module in model.named_modules():

                         if(isinstance(layer_module, th.nn.Conv2d) or isinstance(layer_module,th.nn.Linear)):

		                    #print(name,layer_module)
                            handles.append(layer_module.register_forward_hook(forward_hook(layer)))
                            layer=layer+1

                      model.eval()
                      model(inputs)

                      layer=0
                      for handle in handles:
                         handle.remove()
                      mi_iteration+=1

              current_iteration += 1

     

          with th.no_grad():            

              model.train_accuracy.append((sum(train_acc)*100)/tr_size)
              model.eval()           
              for batch_idx2, (inputs2, targets2) in enumerate(testloader):
                if(batch_idx2==3 and (short_train == True)):
                    break
                inputs2, targets2 = inputs2.cuda(), targets2.cuda()
                output=model(inputs2)
                y_hat = th.argmax(output, 1)
                score = th.eq(y_hat, targets2).sum()
                test_acc.append(score.item())

              model.test_accuracy.append((sum(test_acc)*100)/te_size)
          end=current_iteration

          print(optimizer.param_groups[0]['lr'])
          scheduler.step()
          

          print('---------------Epoch number: {}'.format(ended_epoch+epoch+1),
              '--------Train accuracy: {}'.format(model.train_accuracy[-1]),
              '--------Test accuracy: {}'.format(model.test_accuracy[-1]),'----------------\n')

    model.best_tetr_acc(prunes)
    acc=model.test_accuracy[-1]
    ended_epoch=ended_epoch+epoch+1
    #ended_iteration=end
    
    print("\nAccuracy change : ",ended_accuracy,"----------------->",acc)
    
    #----------------writing data-----------

    
    d=[]
    for i in range(total_convs):
      d.append(model.remaining_filters_each_epoch[i][-1])

    d.append(model.train_accuracy[-1])
    d.append(model.test_accuracy[-1])

    with open(folder_name+'vggPrune.csv', 'a', newline='') as myfile:
          wr = csv.writer(myfile)
          if(prunes==1):
              wr.writerow(("conv1_filters","conv2_filters","train_acc", "test_acc"))
          command=model.get_writerow(total_convs+2)
          eval(command)

    myfile.close()
    #-------------------------end writing data---------------

print(model.MI.size())
print(model.sigmas.size())


#mi = model.MI.cpu().detach().numpy().astype('float16').reshape(1, current_iteration,total_layers, 2)
mi = model.MI.cpu().detach().numpy().astype('float16')
#sigmas=model.sigmas.cpu().detach().numpy().astype('float16').reshape(1,total_layers, current_iteration)
sigmas=model.sigmas.cpu().detach().numpy().astype('float16')
initial_iter=np.int64((tr_size // batch_size_tr)*epochs)
prune_iters=np.int64((tr_size // batch_size_tr)*mi_calc_epochs)
np.savez_compressed(folder_name+'vgg.npz',
                      a=mi, b=sigmas,c=initial_iter,d=prune_iters)
