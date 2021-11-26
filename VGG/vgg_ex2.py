''' 1.Traning: whole train data
    2.Testing: Whole test data batch wise'''
import random
import torch as th
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
#from CNN_Hero import CNN_Hero
from new_VGG16 import VGG16
import csv
from torch.optim.lr_scheduler import MultiStepLR

seed =1787
random.seed(seed)
import os
os.environ['PYTHONHASHSEED'] = str(seed)

th.manual_seed(seed)
th.cuda.manual_seed(seed)
th.cuda.manual_seed_all(seed)
th.backends.cudnn.deterministic = True
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

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

epochs = 300
new_epochs=90
mi_epoch=0
mi_calc_epochs=10

total_layers=15
total_convs=13


prune_limits=[57 , 57 , 128 , 128 ,256,256,256,512,512,512,512,512,512]
#prune_limits=[19,48,65,65,96,112,110,186,79,79,49,48,60]
decision_count=th.ones((total_convs))

tr_size = 50000
te_size=10000

''''----m3 acc and MI with train data----'''
n_iterations = (tr_size // batch_size_tr)*epochs

'''-----m4 acc and MI with test data---'''
#n_iterations = (te_size // batch_size_te)*epochs

activation = 'relu'
short_train=False

#x_tr, y_tr, x_te, y_te = load_cifar10(path, gpu)
#print(x_tr.size(), y_tr.size(), x_te.size(), y_te.size())
#-----------------learning pytorch
#x_tr= x_tr[:300]
#y_tr= y_tr[:300]
#x_te= x_te[:300]
#y_te= y_te[:300]
#tr_size = 300
#te_size=300
#short_train=True
#n_iterations = (tr_size // batch_size_tr)*epochs
#print(n_iterations)

n_iterations = (tr_size // batch_size_tr)*(epochs - mi_epoch)


model=VGG16(10,activation ,n_iterations).cuda()



with th.no_grad():

  folder_name=model.create_folders(total_convs)
  #---------intialize layer numbers and names dictionary------
  model.intialize_layer_name_num(model)

  #---------intialize pruned and remaning filters-----------
  model.filters_in_each_layer(model)
  
  #--------calculate remaining filters in each epoch--------
  model.remaining_filters_per_epoch(model=model,initial=True)

optimizer = th.optim.SGD(model.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[80,140,230], gamma=0.1)

criterion = nn.CrossEntropyLoss()
for n in range(N):

    current_iteration = 0
    mi_iteration=0

    for epoch in range(epochs):

      start=current_iteration


      train_acc=[]

      for batch_num, (inputs, targets) in enumerate(trainloader):

        if(batch_num==3 and short_train==True):
                break

        inputs = inputs.cuda()
        targets = targets.cuda()


        model.train()

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output[0], targets)
        loss.backward()
        optimizer.step()

        with th.no_grad():

          y_hat = th.argmax(output[0], 1)
          score = th.eq(y_hat, targets).sum()
          train_acc.append(score.item())


        if(epoch>=mi_epoch):
           with th.no_grad():
        
              model.compute_mi(inputs,targets , model, gpu, mi_iteration)
              mi_iteration+=1

        current_iteration += 1
      
      with th.no_grad():

        model.train_accuracy.append((sum(train_acc)*100)/tr_size)
            
        test_acc=[]
        for batch_nums, (inputs2, targets2) in enumerate(testloader):
            if(batch_nums==3 and short_train==True):
                break

            inputs, targets = inputs2.cuda(), targets2.cuda()

            test_acc.append(model.predict(inputs,targets, model, gpu))

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
model.test_accuracy.append(final_train_acc)
model.train_accuracy.append(final_test_acc)

#----------------writing data---------------------
    
d=[]
for i in range(total_convs):
      d.append(model.remaining_filters_each_epoch[i][-1])

d.append(model.train_accuracy[-1])
d.append(model.test_accuracy[-1])

with open(folder_name+'vggPrune.csv', 'a', newline='') as myfile:
          wr = csv.writer(myfile)
          wr.writerow([d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13],d[14]])

myfile.close()
#-------------------------end writing data---------------

a=[]

for name, layer_module in model.named_modules():

  if(isinstance(layer_module, th.nn.Conv2d)):
    a.append(layer_module)


for p in range(total_layers):
   model.sigmas_fixed[p]=model.sigmas[p,-1]


decision=True
#----------------------retrain the model--------------------
while(decision==True):
    #----------------------prune the model--------------------


    with th.no_grad():

      #------------coumput IB bounds for each filter in each layer------------

      for batch_num, (inputs, targets) in enumerate(trainloader):
        if(batch_num==3 and short_train==True):
          break
        inputs = inputs.cuda()
        targets = targets.cuda()

        layer_bounds1=model.compute_bounds(inputs, targets, model, gpu,batch_num, current_iteration-1,
total_convs,folder_name,decision_count,no_of_batches=(tr_size // batch_size_tr))


      
      #-------------------pruning each conv layer---------------
      for i in range(len(layer_bounds1)-1,-1,-1):

          if(len(model.remaining_filters[i]) > prune_limits[i] ):
            model.prune_lenet(a[i],layer_bounds1[i],i)
            print(a[i],len(layer_bounds1[i]))
          else:
            print(model.layer_name_num[i],"has reached",prune_limits[i] )
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

    with th.no_grad():
      new_mi_iterations=((tr_size // batch_size_tr)*mi_calc_epochs)
      new_MI = th.zeros((new_mi_iterations,total_layers, 2)).cuda() #-----------change to 16 for vgg
      new_sigmas = th.zeros((total_layers, new_mi_iterations)).cuda() #---------change to 16 for VGG
      model.MI=th.cat((model.MI,new_MI))
      model.sigmas= th.cat((model.sigmas,new_sigmas),dim=1)
    
    #optimizer = th.optim.SGD(model.parameters(), lr=0.01,momentum=0.9, weight_decay=5e-4)
    optimizer.param_groups[0]['lr']=0.01
    scheduler = MultiStepLR(optimizer, milestones=[40,70,90], gamma=0.1)
    
    for epoch in range(new_epochs):

          start=current_iteration
          train_acc=[]
          test_acc=[]

          for batch_num, (inputs, targets) in enumerate(trainloader):

            if(batch_num==3 and short_train==True):
               break

            inputs = inputs.cuda()
            targets = targets.cuda()

            model.train()

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output[0], targets)
            loss.backward()
            optimizer.step()

            with th.no_grad():

               y_hat = th.argmax(output[0], 1)
               score = th.eq(y_hat, targets).sum()

               train_acc.append(score.item())

               if(epoch>=(new_epochs-mi_calc_epochs)):
                   
                        model.compute_mi(inputs,targets , model, gpu, mi_iteration)
                        mi_iteration+=1

               current_iteration += 1           

          with th.no_grad():            

              model.train_accuracy.append((sum(train_acc)*100)/tr_size)                       
              for batch_idx2, (inputs2, targets2) in enumerate(testloader):
                 if(batch_idx2==3 and short_train==True):
                    break
                 inputs, targets = inputs2.cuda(), targets2.cuda()
                 test_acc.append(model.predict(inputs,targets, model, gpu))           
              '''for idx_te in batches_te:

                x_te_b = x_tr[idx_te]
                y_te_b = y_tr[idx_te]
                test_acc.append(model.predict(x_te_b, y_te_b, model, gpu))'''


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
              wr.writerow(("conv1_filters","conv2_filters","conv3_filters","conv4_filters","conv5_filters",
              "conv6_filters","conv7_filters","conv8_filters","conv9_filters","conv10_filters",
              "conv11_filters","conv12_filters","conv13_filters","train_acc", "test_acc"))
          wr.writerow([d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13],d[14]])

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
