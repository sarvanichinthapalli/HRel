import numpy as np
import matplotlib.pyplot as plt
file_name='vgg'


loaded = np.load(file_name+'.npz')
#print(loaded.files)
#print(loaded['a'].shape)


MI=loaded['a']
sigmas=loaded['b']

print(sigmas.shape,'-------',MI.shape)
print(MI.shape[1],'  hdden layers.....',MI.shape[0],'  runs')

#MI=np.mean(MI,axis=0)
hidden_layers=np.size(MI,1)

if hidden_layers==5:
	c_lab =['Reds','Purples','Blues','Greens','Oranges']
	c_plain =['Red','Purple','Blue','Green','Orange']
else:
	c_lab = ['Purples','Purples','Reds','Reds','Greys','Greys','Greys','Greens','Greens','Greens','Oranges','Oranges','Oranges',  'Blues','Blues','Purples']
	c_plain =['Red','Red','Red','Red','Purple','Purple','Purple','Blue','Blue','Blue','Blue','Green','Green','Green','Orange','Orange']

x=np.arange(0,sigmas.shape[1],1)
y=[]
print(sigmas.shape[0])
for i in range(sigmas.shape[0]):
	y.append(sigmas[i,:].tolist())
for i in range(sigmas.shape[0]):
	#if(i not in [0,4,7,10,13]):
	if(i not in [20]):
	   plt.plot(x,y[i],color=c_plain[i],label='layer '+str(i+1))
plt.legend(facecolor='white')
plt.xlabel('iterations',fontsize=16)
plt.ylabel('$sigma$',fontsize=20)
plt.tight_layout()
ax = plt.gca()
ax.axes.xaxis.set_visible(False)

total_n_iter=sigmas.shape[1]
initial_iter=loaded['c']
prune_iters=loaded['d']
plt.axvline(x=initial_iter,linestyle =":",color='gray')
print(initial_iter+1,total_n_iter,prune_iters)
for k in range(initial_iter,total_n_iter,prune_iters):
        print(k)
        plt.axvline(x=k,linestyle =":",color='gray')


plt.tight_layout()
#plt.grid(b=None)                      #-----NO grids----------

ax = plt.gca()
#ax.tick_params(axis='x', labelsize=20)#----x-axis numbers size-----
ax.tick_params(axis='y', labelsize=16)#----x-axis numbers size-----

leggy=ax.legend(loc='upper left',framealpha=5, fontsize='x-large',ncol=2) #------Legend place
leggy.get_frame().set_edgecolor('black') #-----Legend frame color
leggy.get_frame().set_alpha(0.2)
leggy.get_frame().set_facecolor('pink')
#ax.xaxis.label.set_color('black')    #----x-axis numbers color----
#ax.tick_params(axis='x', colors='black') #---x-axis label color---
ax.yaxis.label.set_color('black')    #----y-axis numbers color----
ax.tick_params(axis='y', colors='black') #---y-axis label color---

plt.ylim(0,150) #FOR DEATILED FIGURE

plt.gcf().savefig('plots/'+file_name + "Sigmas_1_2.png")
plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------------

plt.figure(1, figsize=(10, 6))
plt.style.use('ggplot')

MI=MI[:,:,:]
avg_iterations=MI[:,:,:].shape[0]//1000

avg_mi=np.ones((avg_iterations,MI.shape[1],MI.shape[2])) 
print(MI.shape)
#########code for infomration plane
print(avg_mi.shape)

for k in range(avg_iterations):
	avg_mi[k,:,:]= np.mean(MI[k*1000 : (k+1)*1000],axis=0) 
	
print('only intial iterrations before pruning are',initial_iter)

for j in range(hidden_layers):
	if(j not in [20]):
		plt.scatter(avg_mi[:, j, 0],avg_mi[:, j, 1], cmap=c_lab[j], c=np.arange(0, avg_mi.shape[0], 1), edgecolor=c_lab[j][:-1], s=50)

for j in range(hidden_layers):
	if(j not in [20]):
		plt.scatter(avg_mi[-1, j, 0], avg_mi[-1, j, 1], c=c_lab[j][:-1], label='Layer {}'.format(j+1), s=50)


plt.ylabel(r'$I(L_i;Y)$',fontsize=22) #-----y-axis label------
plt.xlabel(r'$I(X;L_i)$', fontsize= 22)#-----x-axis label------

plt.tight_layout()
#plt.grid(b=None)                      #-----NO grids----------

ax = plt.gca()
ax.tick_params(axis='x', labelsize=18)#----x-axis numbers size-----
ax.tick_params(axis='y', labelsize=18)#----x-axis numbers size-----

leggy=ax.legend(loc='upper left',framealpha=5, fontsize='x-large') #------Legend place
leggy.get_frame().set_edgecolor('black') #-----Legend frame color

ax.xaxis.label.set_color('black')    #----x-axis numbers color----
ax.tick_params(axis='x', colors='black') #---x-axis label color---
ax.yaxis.label.set_color('black')    #----y-axis numbers color----
ax.tick_params(axis='y', colors='black') #---y-axis label color---


plt.gcf().savefig('plots/'+file_name + "mi_before_prune.png")
plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------

plt.figure(1, figsize=(10, 6))
plt.style.use('ggplot')

MI=MI[:300*500,:,:] # intial_epochs * no.of minibatches 
avg_iterations=MI.shape[0]//1000 

avg_mi=np.ones((avg_iterations,MI.shape[1],MI.shape[2]))
print(MI.shape)
#########code for infomration plane
print(avg_mi.shape)

for k in range(avg_iterations):
	avg_mi[k,:,:]= np.mean(MI[k*1000 : (k+1)*1000],axis=0) 
	
print(avg_mi.shape)

for j in range(hidden_layers):
	if(j not in [20]):
		plt.scatter(avg_mi[:, j, 0],avg_mi[:, j, 1], cmap=c_lab[j], c=np.arange(0, avg_mi.shape[0], 1), edgecolor=c_lab[j][:-1], s=50)

for j in range(hidden_layers):
	if(j not in [20]):
		plt.scatter(avg_mi[-1, j, 0], avg_mi[-1, j, 1], c=c_lab[j][:-1], label='Layer {}'.format(j+1), s=50)

#plt.legend(facecolor='white')
#plt.ylabel(r'$I(X,h_i)$', fontname="Arial",fontsize=16)

plt.ylabel(r'$I(L_i;Y)$',fontsize=22) #-----y-axis label------
plt.xlabel(r'$I(X;L_i)$', fontsize= 22)#-----x-axis label------

plt.tight_layout()
#plt.grid(b=None)                      #-----NO grids----------

ax = plt.gca()
ax.tick_params(axis='x', labelsize=18)#----x-axis numbers size-----
ax.tick_params(axis='y', labelsize=18)#----x-axis numbers size-----

leggy=ax.legend(loc='upper left',framealpha=5,fontsize='x-large') #------Legend place
leggy.get_frame().set_edgecolor('black') #-----Legend frame color

ax.xaxis.label.set_color('black')    #----x-axis numbers color----
ax.tick_params(axis='x', colors='black') #---x-axis label color---
ax.yaxis.label.set_color('black')    #----y-axis numbers color----
ax.tick_params(axis='y', colors='black') #---y-axis label color---

plt.gcf().savefig('plots/'+file_name + "complete_MI.png")
plt.show()


################code for histogram
'''
temp=MI[10:60,:,:] # index 0= iterations , index1=layer number
print(temp.shape)

#ckecking for nan values 
for k in range(16):
	temp1=temp[:,k,1]
	array_sum = np.sum(temp1)
	array_has_nan = np.isnan(array_sum)
	print('layer ',str(k),array_has_nan)
	print(temp1)

#callulating information bottleneck 

last_iterations=temp.shape[0]
s=[]
for k in range(last_iterations):
	s.append(temp[k,0] - (2* temp[k,1]))
print(len(s))
#fig, ax = plt.subplots(figsize =(10, 7)) 
bins,n=np.histogram(s,bins=40)
left = np.array(bins[:-1])
right = np.array(bins[1:])
bottom = np.zeros(len(left))
top = bottom + n


# we need a (numrects x numsides x 2) numpy array for the path helper
# function to build a compound path
XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

# get the Path object
barpath = path.Path.make_compound_path_from_polys(XY)

# make a patch out of it
patch = patches.PathPatch(barpath)
ax.add_patch(patch)

# update the view limits
ax.set_xlim(left[0], right[-1])
ax.set_ylim(bottom.min(), top.max())

plt.show()

plt.scatter(MI[:,12,0],MI[:,12,1])
plt.show() 

'''
