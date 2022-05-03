
# coding: utf-8

# In[ ]:


#get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import numpy as np
import pickle
import os
import time

      
epoch = 44
exp = 'cifar10'
# exp = 'MNIST'

N = 87197 # image number

y_train = np.load('/local/scratch/jrs596/ResNetVAE/results_152_ForesArabData_356_LatentDim/y_{}_train_epoch{}.npy'.format(exp, epoch))
z_train = np.load('/local/scratch/jrs596/ResNetVAE/results_152_ForesArabData_356_LatentDim/z_{}_train_epoch{}.npy'.format(exp, epoch))
#classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  # cifar10
# classes = np.arange(10) #MNIST
classes = tuple(os.listdir('/local/scratch/jrs596/dat/ResNetFung50+_images_organised/train'))
#classes = ['PotatoesDiseased', 'PeachesHealthy', 'TomatoesHealthy']
# ## Direct projection of latent space

# In[ ]:


y_train = y_train[:N]
z_train = z_train[:N]

fig = plt.figure(figsize=(12, 10))
plots = []
#markers = ['o', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']

for i, c in enumerate(classes):
    ind = (y_train == i).tolist() or ([j < N // len(classes) for j in range(len(y_train))])
    color = cm.jet([i / len(classes)] * sum(ind))
    plots.append(plt.scatter(z_train[ind, 1], z_train[ind, 2], c=color, s=8, label=i))

plt.axis('off')
plt.legend(plots, classes, fontsize=14, loc='upper right')
plt.title('{} (direct projection: {}-dim -> 2-dim)'.format(exp, z_train.shape[1]), fontsize=14)
plt.savefig("./ResNetVAE_{}_direct_plot.png".format(exp), bbox_inches='tight', dpi=600)
plt.show()

# ## Use t-SNE for dimension reduction

# ### compressed to 2-dimension

# In[ ]:


z_embed = TSNE(n_components=2, n_iter=12000).fit_transform(z_train[:N])


# In[ ]:


fig = plt.figure(figsize=(12, 10))
plots = []
markers = ['o', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']  # select different markers
for i, c in enumerate(classes):
    ind = (y_train[:N] == i).tolist()
    color = cm.jet([i / len(classes)] * sum(ind))
    # plot each category one at a time 
    plots.append(plt.scatter(z_embed[ind, 0], z_embed[ind, 1], c=color, s=8, label=i))

plt.axis('off')
plt.xlim(-150, 150)
plt.ylim(-150, 150)
plt.legend(plots, classes, fontsize=14, loc='upper right')
plt.title('{} (t-SNE: {}-dim -> 2-dim)'.format(exp, z_train.shape[1]), fontsize=14)
plt.savefig("./ResNetVAE_{}_embedded_plot.png".format(exp), bbox_inches='tight', dpi=600)
plt.show()


# ### compressed to 3-dimension

# In[ ]:


z_embed3D = TSNE(n_components=3, n_iter=12000).fit_transform(z_train[:N])


# In[ ]:


fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

plots = []
markers = ['o', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']  # select different markers
for i, c in enumerate(classes):
    ind = (y_train[:N] == i).tolist()
    color = cm.jet([i / len(classes)] * sum(ind))
    # plot each category one at a time 
    ax.scatter(z_embed3D[ind, 0], z_embed3D[ind, 1], c=color, s=8, label=i)

ax.axis('on')

r_max = 20
r_min = -r_max

ax.set_xlim(r_min, r_max)
ax.set_ylim(r_min, r_max)
ax.set_zlim(r_min, r_max)
ax.set_xlabel('z-dim 1')
ax.set_ylabel('z-dim 2')
ax.set_zlabel('z-dim 3')
ax.set_title('{} (t-SNE: {}-dim -> 3-dim)'.format(exp, z_train.shape[1]), fontsize=14)
ax.legend(plots, classes, fontsize=14, loc='upper right')
plt.savefig("./ResNetVAE_{}_embedded_3Dplot.png".format(exp), bbox_inches='tight', dpi=600)
plt.show()

