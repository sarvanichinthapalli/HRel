''' 1.Traning: whole train data
    2.Testing: Whole test data batch wise'''
import random
import torch as th
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from new_Resnet import resnet_50
import csv
from itertools import zip_longest
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from data import imagenet
#from data import imagenet_dali 
import utils.common as utils

seed =1787
random.seed(seed)
import os
os.environ['PYTHONHASHSEED'] = str(seed)

th.manual_seed(seed)
th.cuda.manual_seed(seed)
th.cuda.manual_seed_all(seed)
th.backends.cudnn.deterministic = True

cudnn.benchmark = True
cudnn.enabled=True


N = 1
import sys
short=False

batch_size_tr =80
batch_size_te =batch_size_tr
batch_size=batch_size_tr
data_dir='/home/kishank/Documents/Imagenet'

use_dali=False
num_gpu= 2    

epochs = 3
mi_epoch=2
new_epochs=33
mi_calc_epochs=1

total_layers=50+4
total_convs=32 #first conv not included
total_blocks=4



noted_filters=[0,6,14,26]

gpu=True

print('==> Preparing data..')
if use_dali:
        def get_data_set(type='train'):
            if type == 'train':
                return imagenet_dali.get_imagenet_iter_dali('train', data_dir, batch_size,
                                                            num_threads=4, crop=224, device_id=0, num_gpus=num_gpu)
            else:
                return imagenet_dali.get_imagenet_iter_dali('val', args.data_dir, batch_size,
                                                            num_threads=4, crop=224, device_id=0, num_gpus=num_gpu)
        train_loader = get_data_set('train')
        val_loader = get_data_set('val')
        
else:
        data_tmp = imagenet.Data(gpu,data_dir,batch_size)
        train_loader = data_tmp.train_loader
        val_loader = data_tmp.test_loader

        data_tmp2 = imagenet.Data(gpu,data_dir,128)
        train_loader2 = data_tmp2.train_loader
        val_loader2 = data_tmp2.test_loader



if use_dali:
        n_iterations_per_epoch= train_loader._size // batch_size
        n_iterations = (train_loader._size // batch_size)*(epochs - mi_epoch)
else:
        n_iterations_per_epoch= len(train_loader)
        n_iterations = len(train_loader)*(epochs - mi_epoch)

batch_size=batch_size_tr



#___________________CREATE MODEL AND LOAD PRETRAINED WEIGHTS_____________________
if gpu:
    model=resnet_50(n_iterations*num_gpu,[0.0]*50).to('cuda:0')
else:
    model=resnet_50(n_iterations*num_gpu,[0.0]*50)

pretrained_model = models.resnet50(pretrained=True).to('cuda:0')
state = {'model': pretrained_model.state_dict()}
th.save(state,'pretrained_res50_model.pth')
checkpoint = th.load('pretrained_res50_model.pth')
model.load_state_dict(checkpoint['model'])



def train(epoch, train_loader, model, criterion, optimizer,scheduler,mi_iter,sigma_update):

    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    model.train()
    if use_dali:
        num_iter = train_loader._size // batch_size
        print('________',train_loader._size , batch_size)
    else:
        num_iter = len(train_loader)

    print_freq=1000
    mi_iteration=mi_iter

    for batch_idx, batch_data in enumerate(train_loader):

        
        #if(batch_idx==4): #_________________________COMMENT THIS
            #import sys
            #sys.exit()
            #del batch_data      
            #break
        if use_dali:
            images = batch_data[0]['data'].cuda()
            targets = batch_data[0]['label'].squeeze().long().cuda()
        else:
            images = batch_data[0].to('cuda:0')
            targets = batch_data[1].to('cuda:0')

        #if(epoch < mi_epoch):
        logits = model(images)
        loss = criterion(logits, targets)

        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
        n = images.size(0)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #___________________________hooks code should start from here______________

        if(epoch>=mi_epoch):

            #if(sigma_update==False):
            
            #  batch_limit=new_mi_iterations
            #  if(batch_idx==batch_limit):
            #    print('batch lmit..',batch_limit,' batch_id=',batch_idx,'mi_iteration=',mi_iteration)
            #    break
            
            with th.no_grad():
              
              y=model.module.one_hot(targets, True)
              if(num_gpu==2):
                  k_y=[]
                  k_x=[]
                  index=int((n+1)/2)
                  #print("index=",index)
                  k_y.append(model.module.kernel_mat(model,y[:index], [], sigma=th.tensor(0.1).to('cuda:0')))
                  k_y.append(model.module.kernel_mat(model,y[index:], [], sigma=th.tensor(0.1).to('cuda:0')))
                  k_x.append(model.module.kernel_mat(model,images[:index], [], sigma=th.tensor(8.0).to('cuda:0')))
                  k_x.append(model.module.kernel_mat(model,images[index:], [], sigma=th.tensor(8.0).to('cuda:0')))
              else:
                  k_y = model.module.kernel_mat(model,y, [], sigma=th.tensor(0.1).to('cuda:0'))
                  k_x = model.module.kernel_mat(model,images, [], sigma=th.tensor(8.0).to('cuda:0'))

              if(sigma_update==True):

                 def forward_hook(layer_name):
                         def hook(module, input, output):
                             if(num_gpu==2):
                                if(str(output.device)=='cuda:0'):
                                   model.module.cal_mi(output,k_x[0], k_y[0], model,layer_name, mi_iteration,sigma_update)
                                   #print(layer_name,'~~~~~~~~',output.size(),'~~~~~~~',output.device,'~~~~',mi_iteration)
                                elif(str(output.device)=='cuda:1'):
                                   model.module.cal_mi(output,k_x[1], k_y[1], model,layer_name, mi_iteration+1,sigma_update)
                                   #print(layer_name,'~~~~~~~~',output.size(),'~~~~~~~',output.device,'~~~~',mi_iteration)
                             else:
                                #print(layer_name,'~~~~~~~~',output.size(),'~~~~~~~',output.device,'~~~~',mi_iteration)
                                model.module.cal_mi(output,k_x, k_y, model,layer_name, mi_iteration,sigma_update)


                         return hook
              
                 model.handles=[]

                 for name, layer_module in model.named_modules():
                     if(isinstance(layer_module, th.nn.Conv2d) and name!='module.conv1' and name.find('conv2')!=-1 and name.find('downsample')==-1):
                       #print(name)
                       model.handles.append(layer_module.register_forward_hook(forward_hook(name)))

              def forward_hook(layer_name):
                         def hook(module, input, output):
                             if(num_gpu==2):
                                if(str(output.device)=='cuda:0'):
                                   model.module.cal_mi_durga(output,k_x[0], k_y[0], model,layer_name, mi_iteration,sigma_update)
                                   #print(layer_name,'~~~~~~~~',output.size(),'~~~~~~~',output.device,'~~~~',mi_iteration)
                                elif(str(output.device)=='cuda:1'):
                                   model.module.cal_mi_durga(output,k_x[1], k_y[1], model,layer_name, mi_iteration+1,sigma_update)
                                   #print(layer_name,'~~~~~~~~',output.size(),'~~~~~~~',output.device,'~~~~',mi_iteration)
                             else:
                                #print(layer_name,'~~~~~~~~',output.size(),'~~~~~~~',output.device,'~~~~',mi_iteration)
                                model.module.cal_mi_durga(output,k_x, k_y, model,layer_name, mi_iteration,sigma_update)


                         return hook
              
              model.durga_handles=[]

              for name, layer_module in model.named_modules():
                     if(isinstance(layer_module, th.nn.Conv2d) and name!='module.conv1' and name.find('conv1')!=-1 and name.find('downsample')==-1):
                       #print('*******',name)
                       model.durga_handles.append(layer_module.register_forward_hook(forward_hook(name)))
              

              model.eval()
              
              model(images) #hooks called for each of the



              if(sigma_update==True):
                 for handle in model.handles:
                   handle.remove()

              for handle in model.durga_handles:
                   handle.remove()

              mi_iteration+=num_gpu #+2 if 2 GPUs used o.w 1
              #print(mi_iteration,'*****')
              del images
              del targets


          #___________hooks end here_____________
        #if(True):
        if batch_idx % print_freq == 0:
                logger.info(
                    'Epoch[{0}]({1}/{2}): '
                    'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                        epoch, batch_idx, num_iter,
                        top1=top1, top5=top5))

        

    scheduler.step()
    print(optimizer.param_groups[0]['lr'])
    return top1.avg, top5.avg,mi_iteration

def validate(epoch, val_loader, model, criterion):

    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    if use_dali:
        num_iter = val_loader._size // batch_size
    else:
        num_iter = len(val_loader)

    model.eval()
    with th.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            #if(batch_idx==4):
                #break
            if use_dali:
                images = batch_data[0]['data'].cuda()
                targets = batch_data[0]['label'].squeeze().long().cuda()
            else:
                images = batch_data[0].cuda()
                targets = batch_data[1].cuda()

            logits = model(images)
            loss = criterion(logits, targets)

            pred1, pred5 = utils.accuracy(logits, targets, topk=(1, 5))
            n = images.size(0)
            top1.update(pred1.item(), n)
            top5.update(pred5.item(), n)
            
    logger.info('* Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                    .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


optimizer = th.optim.SGD(model.parameters(), lr=0.0001, weight_decay=2e-4,momentum=0.9)
scheduler = MultiStepLR(optimizer, milestones=[5], gamma=0.1)

criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()
criterion_smooth = utils.CrossEntropyLabelSmooth(1000, 0.1)
criterion_smooth = criterion_smooth.cuda()

#__________MAKE MODEL PARALLEL___________________

model = th.nn.DataParallel(model, device_ids=[0,1]).to('cuda')

#_________ATTRIBUTES OF PARALLEL_______________



model.sigmas = th.zeros((32, n_iterations*num_gpu)).cuda()

#model.down_sigmas = th.zeros((4, n_iterations*num_gpu)).cuda()
model.MI = th.zeros((n_iterations*num_gpu,16,2)).cuda()

model.sigmas_fixed=th.zeros((32)).cuda()
#model.down_sigmas_fixed=th.zeros((4)).cuda()

model.ignore_iterations=[]

model.layer_name_num={}
model.MI_layer_name_num={}
#model.down_layer_name_num={}

model.pruned_filters={}
model.remaining_filters={}
model.remaining_filters_each_epoch=[]
        
model.sub_MI_X=[]
model.sub_MI_Y=[]

model.module.intialize_layer_name_num(model)
model.module.intialize_MI_layer_name_num(model)
#model.module.intialize_down_layer_name_num(model)
#print(model.layer_name_num)
#print(model.MI_layer_name_num)


with th.no_grad():

  folder_name=model.module.create_folders(total_convs)

  logger=utils.get_logger(folder_name+'logger.log')

  #---------intialize layer numbers and names dictionary------
  #model.module.intialize_layer_name_num(model)

  #---------intialize pruned and remaning filters-----------
  model.module.filters_in_each_layer(model)
  
  #--------calculate remaining filters in each epoch--------
  model.module.remaining_filters_per_epoch(model=model,initial=True)
mi_iteration=0

for n in range(N):

    
    mi_iteration=0

    for epoch in range(epochs):
        train_top1_acc,  train_top5_acc,mi_iteration = train(epoch,  train_loader, model, criterion_smooth, optimizer,scheduler,mi_iteration,True)
        valid_top1_acc, valid_top5_acc = validate(epoch, val_loader2, model, criterion)

        if use_dali:
            train_loader.reset()
            val_loader.reset()
        
        state = {'model': model.module.state_dict(),
               'optimizer':optimizer.state_dict(),
               'scheduler':scheduler.state_dict(),
              'sigmas':model.sigmas,
              'MI':model.MI,
              'epoch':epoch,
              'mi_iteration':mi_iteration,
              'te_acc':[valid_top1_acc, valid_top5_acc],
              'tr_acc':[train_top1_acc, train_top5_acc]

            }
        th.save(state,folder_name+'stage1.pth')

        
ended_epoch=epoch

prunes=0

#----------------writing data---------------------
a=[]
for layer_name, layer_module in model.named_modules():

  if(isinstance(layer_module, th.nn.Conv2d) and layer_name != "module.conv1" and layer_name.find('conv3')==-1 and layer_name.find('downsample')==-1):
    a.append(layer_module)

d=[]
for i in range(total_convs):
      #print(model.remaining_filters_each_epoch[i])
      if(i in noted_filters):
        d.append(model.remaining_filters_each_epoch[i][-1])

d.append(train_top1_acc)
d.append(valid_top1_acc)
d.append(train_top5_acc)
d.append(valid_top5_acc)


with open(folder_name+'resnet50.csv', 'a', newline='') as myfile:
          wr = csv.writer(myfile)

          command=model.module.get_writerow(len(noted_filters)+4)
          eval(command)
          #wr.writerow([d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13],d[14],d[15],d[16]])

myfile.close()

for p in range(total_convs):
    model.sigmas_fixed[p]=th.mean(model.sigmas[p,-500:])


print('yes finished learning sigmas')

#___________PRUNING ITERATIONS START____________________

while(prunes<9):

    with th.no_grad():


      #___________hook creation start______________

      def forward_hook(layer_name):
                      def hook(module, input, output):
                          gpu_id=str(output.device)[-1]
                          model.module.compute_bounds(model,output,layer_name,num_gpu,gpu_id,logger)
                      return hook
              
      handles=[]

      for name, layer_module in model.named_modules():

          if(isinstance(layer_module, th.nn.Conv2d) and name != "module.conv1" and name.find('conv3')==-1 and name.find('downsample')==-1 ):

             handles.append(layer_module.register_forward_hook(forward_hook(name)))

      #___________hook creation end______________
      
      z=[]
      small_z=[]

      #_________NEW DATALOADER________________
      data_tmp3 = imagenet.Data(gpu,data_dir,256)
      train_loader3 = data_tmp3.train_loader

      no_of_batches=len(train_loader3)//4
      #no_of_batches=2

      for batch_num, batch_data in enumerate(train_loader3):
        if(batch_num%100==0):
           logger.info('computing_bounds of batch='+str(batch_num)+'/'+str(no_of_batches))

        if(batch_num==no_of_batches):
           break

        if use_dali:
            images = batch_data[0]['data'].cuda()
            targets = batch_data[0]['label'].squeeze().long().cuda()
        else:
            images = batch_data[0].to('cuda:0')
            targets = batch_data[1].to('cuda:0')

       
        if(batch_num==0):  #______INTIALIZE SUB_MI_Y_____

            #layer=0
            for name, layer_module in model.named_modules():

                if(isinstance(layer_module, th.nn.Conv2d) and name != "module.conv1"  and name.find('conv3')==-1 and name.find('downsample')==-1  ):
                    
                    filters_num=model.remaining_filters[model.layer_name_num[name]]
                    #print(name, len(filters_num))
                    model.sub_MI_Y.append(th.zeros((no_of_batches*num_gpu,len(filters_num))).cuda())
                    #layer=layer+1

        n = images.size(0)
        y=model.module.one_hot(targets, gpu)
        index=int(n/num_gpu)

        if(num_gpu==2):
                  #print("index=",index)
                  model.k_y=[]
                  model.k_y.append(model.module.kernel_mat(model,y[:index], [], sigma=th.tensor(0.1).to('cuda:0')))
                  model.k_y.append(model.module.kernel_mat(model,y[index:], [], sigma=th.tensor(0.1).to('cuda:0')))
        else:
                  model.k_y = model.kernel_mat(model,y, [], sigma=th.tensor(0.1).to('cuda:0'))

        model.batch_number=batch_num #____update ongoing batch number___

        model.eval()
        model(images) #_____hook called here and bounds calculated_____

        #logger.info('batch {0}/{1}'.format(batch_num,no_of_batches))

        state = {'model': model.module.state_dict(),
               #'optimizer':optimizer.state_dict(),
               #'scheduler':scheduler.state_dict(),
              'sigmas':model.sigmas,
              'MI':model.MI,
              'sub_MI_Y':model.sub_MI_Y,
              'batch_num':batch_num,
              'epoch':ended_epoch,
              'prunes':prunes,
              'fixed_sigmas':model.sigmas_fixed,
              'pruned_filters':model.pruned_filters,
              'remaining_filters':model.remaining_filters,
              'remaining_filters_per_epoch':model.remaining_filters_each_epoch,
              'mi_iteration':mi_iteration,
              'no_of_batches':no_of_batches,
              'batch_size':n, 
              'te_acc':[valid_top1_acc, valid_top5_acc],
              'tr_acc':[train_top1_acc, train_top5_acc]

            }
        th.save(state,str(folder_name)+'stage3_prunes'+str(prunes)+'.pth')
        
 
        if(batch_num+1 == no_of_batches): #--------------create plots for data--------------

          l1norm=[]
          print('Now saving the results...')

          for i in range(total_convs):
              z.append((th.sum(model.sub_MI_Y[i],dim=0)).div(no_of_batches*num_gpu).tolist())
             
          for i in range(total_convs):
            plt.hist(z[i], weights=np.ones(len(z[i])) / len(z[i]))
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
            plt.savefig(folder_name+"distributions/conv"+
            str(i+1)+"/epoch"+str(ended_epoch)+"_"+str(len(model.remaining_filters[i]))+".png")
            plt.clf()
            plt.cla()
            plt.close()

            fig, ax = plt.subplots(figsize=(60,10))
            pcm = ax.pcolormesh(model.sub_MI_Y[i].cpu().numpy())
            fig.colorbar(pcm, ax=ax)
            plt.savefig(folder_name+"mi_values/conv"+
            str(i+1)+"/epoch"+str(ended_epoch)+"_"+str(len(model.remaining_filters[i]))+".png")
            
            plt.close()

          #import sys
          #sys.exit()
          model.sub_MI_X=[]
          model.sub_MI_Y=[]

      #____________END of DataLoader___________

      layer_bounds1=z

      #_____________PRUNING______________________

      for i in range(len(layer_bounds1)-1,-1,-1):
          print('the layer is...',i)

          handles[i].remove()
	
          model.module.prune_filters(model,a[i],layer_bounds1[i],i)
  
 
      prunes+=1

    model.module.remaining_filters_per_epoch(model=model)
    #-----------------------------m3--------------
    new_iterations=(n_iterations_per_epoch)*new_epochs
    #----------------------------m4----------------

    print('new-model starts... ended epoch=',ended_epoch)
    
    with th.no_grad():

      mi_epoch=new_epochs-mi_calc_epochs  #_____UPDATE mi_epoch _______
      new_mi_iterations=((n_iterations_per_epoch)*mi_calc_epochs)

      if(mi_calc_epochs>0):
          print(model.MI.size(),new_mi_iterations)
          new_MI = th.zeros((new_mi_iterations*num_gpu,16, 2)).cuda()
          model.MI=th.cat((model.MI,new_MI)) 
          print(model.MI.size())
        

    
    optimizer = th.optim.SGD(model.parameters(), lr=0.001,momentum=0.9, weight_decay=2e-4) 
    scheduler = MultiStepLR(optimizer, milestones=[10,25], gamma=0.1)

    for epoch in range(new_epochs):

        train_top1_acc,  train_top5_acc,mi_iteration = train(epoch,  train_loader, model, criterion_smooth, optimizer,scheduler,mi_iteration,False)
        valid_top1_acc, valid_top5_acc = validate(epoch, val_loader2, model, criterion)
        if use_dali:
            train_loader.reset()
            val_loader.reset()
                
        state = {'model': model.module.state_dict(),
               'optimizer':optimizer.state_dict(),
               'scheduler':scheduler.state_dict(),
              'sigmas':model.sigmas,
              #'down_sigmas':model.down_sigmas,
              'MI':model.MI,
              'epoch':epoch,
              'ended_epoch':ended_epoch,
              'prunes':prunes,
              'mi_iteration':mi_iteration,
              'fixed_sigmas':model.sigmas_fixed,
              #'down_fixed_sigmas':model.down_sigmas_fixed,
              'pruned_filters':model.pruned_filters,
              'remaining_filters':model.remaining_filters,
              'remaining_filters_per_epoch':model.remaining_filters_each_epoch,
              'te_acc':[valid_top1_acc, valid_top5_acc],
              'tr_acc':[train_top1_acc, train_top5_acc]

            }
        th.save(state,str(folder_name)+'stage2_prune'+str(prunes)+'.pth')
        

    ended_epoch=ended_epoch+new_epochs #____UPDATE ended_epoch_______
    
    #_________writing data________    
    d=[]
    for i in range(total_convs):
          if(i in noted_filters):
              d.append(model.remaining_filters_each_epoch[i][-1])

    d.append(train_top1_acc)
    d.append(valid_top1_acc)
    d.append(train_top5_acc)
    d.append(valid_top5_acc)

    with open(folder_name+'resnet50.csv', 'a', newline='') as myfile:
              wr = csv.writer(myfile)

              command=model.module.get_writerow(len(noted_filters)+4)
              eval(command)
              #wr.writerow([d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13],d[14],d[15],d[16]])

    myfile.close()

    #________end writing data________

from zipfile import ZipFile
import os,glob

directory = os.path.dirname(os.path.realpath(__file__)) #____location current runnig file____
file_paths = []
os.chdir(directory)
for filename in glob.glob("*.py"):
	filepath = os.path.join(directory, filename)
	file_paths.append(filepath)


print('Following files will be zipped:')
for file_name in file_paths:
	print(file_name)
saving_loc = folder_name #location of results
os.chdir(saving_loc)

# writing files to a zipfile
with ZipFile('python_files.zip','w') as zip:

	# writing each file one by one
	for file in file_paths:
		zip.write(file)

mi = model.MI.cpu().detach().numpy().astype('float16')
#down_sigmas=model.down_sigmas.cpu().detach().numpy().astype('float16')
sigmas=model.sigmas.cpu().detach().numpy().astype('float16')
initial_iter=n_iterations*num_gpu
prune_iters=new_mi_iterations*num_gpu
np.savez_compressed(folder_name+'resnet50.npz',a=mi, b=sigmas,c=initial_iter,d=prune_iters)
