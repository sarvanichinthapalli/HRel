''' 1.Traning: whole train data
    2.Testing: Whole test data batch wise'''
import random
import torch as th
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
#from CNN_Hero import CNN_Hero
from new_Resnet import resnet_56
import csv
from itertools import zip_longest
import matplotlib.pyplot as plt
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


th.cuda.set_device(1)
gpu = th.cuda.is_available()
if gpu:

    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = th.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = th.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=2) 

N = 1

batch_size_tr = 100
batch_size_te = 100

epochs = 182
new_epochs=100
mi_epoch=0
mi_calc_epochs=10

total_layers=56
total_convs=54 #first conv not included
total_blocks=3

prune_limits=[8,15,30]

decision_count=th.ones((total_convs))

tr_size = 50000
te_size=10000


activation = 'relu'

#tr_size = 30
#te_size=30
remove=False
n_iterations_per_epoch= (tr_size // batch_size_tr)
n_iterations = (tr_size // batch_size_tr)*(epochs - mi_epoch)

print('n_iterations.....',n_iterations)

if gpu:
    model=resnet_56(n_iterations).cuda()
else:
    model=resnet_56(n_iterations)

count=0

for name, layer_module in model.named_modules():

                  if((isinstance(layer_module, th.nn.ReLU) and name.find("relu2")!= -1) or isinstance(layer_module,th.nn.Linear)):
                       count=count+1
ip_layers=count

'''for name, layer_module in model.named_modules():

		print(name,layer_module)
		print('-----------------------------------',count)
		count=count+1
print(count)
import sys
sys.exit()'''
with th.no_grad():

  folder_name=model.create_folders(total_convs)
  #---------intialize layer numbers and names dictionary------
  model.intialize_layer_name_num(model)

  #---------intialize pruned and remaning filters-----------
  model.filters_in_each_layer(model)
  
  #--------calculate remaining filters in each epoch--------
  model.remaining_filters_per_epoch(model=model,initial=True)


optimizer = th.optim.SGD(model.parameters(), lr=0.1,momentum=0.9, weight_decay=2e-4,nesterov=True)
#scheduler = MultiStepLR(optimizer, milestones=[80,150,240], gamma=0.1)
scheduler = MultiStepLR(optimizer, milestones=[91,136], gamma=0.1)


criterion = nn.CrossEntropyLoss()
for n in range(N):

    current_iteration = 0
    mi_iteration=0

    for epoch in range(epochs):

      start=current_iteration


      train_acc=[]

      for batch_num, (inputs, targets) in enumerate(trainloader):

        if(batch_num==3 and remove==True):
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

          #---------------------------hooks code should start from here-----------
          if(epoch>=mi_epoch):
            if(mi_iteration%100==0):
               print("you entered epoch",epoch,'iterations ',current_iteration%600)
            
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

                  if((isinstance(layer_module, th.nn.Conv2d) and name != "conv1") ):

                    #print(name,layer_module)
                    handles.append(layer_module.register_forward_hook(forward_hook(layer)))
                    layer=layer+1

              model.eval()
              model(inputs) #--------hooks called for each of the

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
            if(batch_nums==3  and remove==True):
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
      #if(epoch%20==0):
      #  optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr'] *(0.5 ** (epoch // lr_drop))
      scheduler.step()
      print(optimizer.param_groups[0]['lr'])
      #scheduler.step()
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
for i in range(total_blocks):
      d.append(model.remaining_filters_each_epoch[i][-1])

d.append(model.train_accuracy[-1])
d.append(model.test_accuracy[-1])

with open(folder_name+'vggPrune.csv', 'a', newline='') as myfile:
          wr = csv.writer(myfile)

          command=model.get_writerow(3+2)
          eval(command)

myfile.close()

a=[]

for layer_name, layer_module in model.named_modules():

  if(isinstance(layer_module, th.nn.Conv2d) and layer_name != "conv1"):
    a.append(layer_module)

#print(a)


for p in range(total_convs):
    model.sigmas_fixed[p]=model.sigmas[p,-1]



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

          if(isinstance(layer_module, th.nn.Conv2d) and name != "conv1"):

             handles.append(layer_module.register_forward_hook(forward_hook(layer)))
             layer=layer+1

      #-----------------------hook creation end -----------------
      
      no_of_batches=(tr_size // batch_size_tr)

      z=[]
      small_z=[]
      model.eval()

      for batch_num, (inputs, targets) in enumerate(trainloader):

        if(batch_num==3  and remove==True):
          break
        #print('batch number...',batch_num)
        inputs = inputs.cuda()
        targets = targets.cuda()
        
        if(batch_num==0):  #------------------Intilaize sub_mi_y if first batch------------------

            layer=0
            for name, layer_module in model.named_modules():

                if(isinstance(layer_module, th.nn.Conv2d) and name != "conv1"):

                    filters_num=model.remaining_filters[layer]
                    model.sub_MI_Y.append(th.zeros((no_of_batches,len(filters_num))).cuda())
                    layer=layer+1

        y=model.one_hot(targets, gpu)
        model.k_y=model.kernel_mat(y, [], sigma=th.tensor(0.1)) #---update kernel matrix for y
        model.batch_num=batch_num                               #---update batch number
        
        
        #----------remove hooks for layers that reached maximum to avoid computing bounds-----------------
        layer=0
        for name, layer_module in model.named_modules():
          if(isinstance(layer_module, th.nn.Conv2d) and name != "conv1"):
              if(len(model.remaining_filters[layer]) <= prune_limits[int(layer/18)] ):
                  handles[layer].remove()
              layer=layer+1

                

        model(inputs) #---------hook called here and bounds calculated 
 
        if(batch_num+1 == no_of_batches): #--------------create plots for data--------------

          beta=2
            
          for i in range(total_convs):            	

              z.append((th.sum(model.sub_MI_Y[i],dim=0)).div(no_of_batches).tolist())

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


      '''layer=0
      for name, layer_module in model.named_modules():
          if(isinstance(layer_module, th.nn.Conv2d) and name != "conv1"):
              print('layer',layer,'..',layer_module._forward_hooks)
              layer=layer+1'''
      for i in range(len(layer_bounds1)-1,-1,-1):
          #print('the layer is...',i)
          handles[i].remove()      
      #-------------------pruning each conv layer and removing hooks---------------
      for i in range(len(layer_bounds1)-1,-1,-1):
          #print('the layer is...',i)
          #handles[i].remove()
          if(len(model.remaining_filters[i]) > prune_limits[int(i/18)] ):
            model.prune_lenet(a[i],layer_bounds1[i],i,prune_limits[int(i/18)] )
          else:
            print(model.layer_name_num[i],"has reached",prune_limits[int(i/18)] )
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
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('new-model starts....for ',new_iterations,' iterations ,ended epoch=',ended_epoch)
    '''layer=0
    for name, layer_module in model.named_modules():
          if(isinstance(layer_module, th.nn.Conv2d) and name != "conv1"):
              print('layer',layer,'..',layer_module._forward_hooks)
              layer=layer+1'''

    with th.no_grad():
      new_mi_iterations=((tr_size // batch_size_tr)*mi_calc_epochs)
      new_MI = th.zeros((new_mi_iterations,total_convs, 2)).cuda() 
      new_sigmas = th.zeros((total_convs, new_mi_iterations)).cuda() 
      model.MI=th.cat((model.MI,new_MI))
      model.sigmas= th.cat((model.sigmas,new_sigmas),dim=1)

      new_MI = th.zeros((new_mi_iterations,ip_layers, 2)).cuda() 
      new_sigmas = th.zeros((ip_layers, new_mi_iterations)).cuda() 
    
    #optimizer = th.optim.SGD(model.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
    optimizer.param_groups[0]['lr']=0.01
    scheduler = MultiStepLR(optimizer, milestones=[20,70], gamma=0.1)
    
    for epoch in range(new_epochs):

          start=current_iteration
          train_acc=[]
          test_acc=[]

          for batch_num, (inputs, targets) in enumerate(trainloader):

            if(batch_num==3  and remove==True):
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
                      #print('mi=',mi_iteration,'cure=',current_iteration,'epoch=',epoch)
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

                         if((isinstance(layer_module, th.nn.Conv2d)  and name != "conv1")):

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
                if(batch_idx2==3  and remove==True):
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
    for i in range(total_blocks):
      d.append(model.remaining_filters_each_epoch[i][-1])

    d.append(model.train_accuracy[-1])
    d.append(model.test_accuracy[-1])

    with open(folder_name+'vggPrune.csv', 'a', newline='') as myfile:
          wr = csv.writer(myfile)
          if(prunes==1):
              wr.writerow(("block1_filters","block2_filters","block3_filters","train_acc", "test_acc"))
          command=model.get_writerow(3+2)
          eval(command)

    myfile.close()
    #-------------------------end writing data---------------





print(model.MI.size())
print(model.sigmas.size())



mi = model.MI.cpu().detach().numpy().astype('float16')
sigmas=model.sigmas.cpu().detach().numpy().astype('float16')
initial_iter=np.int64((tr_size // batch_size_tr)*epochs)
prune_iters=np.int64((tr_size // batch_size_tr)*mi_calc_epochs)
np.savez_compressed(folder_name+'vgg.npz',
                      a=mi, b=sigmas,c=initial_iter,d=prune_iters)
