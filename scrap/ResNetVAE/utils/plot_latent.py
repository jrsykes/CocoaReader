
# coding: utf-8

# In[ ]:


#get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import pickle
import os
import time

      
y_train = np.load('/local/scratch/jrs596/ResNetVAE/ForesArabData_Random_PredictedZ/y_cifar10_train_epoch.npy')
z_train = np.load('/local/scratch/jrs596/ResNetVAE/ForesArabData_Random_PredictedZ/z_cifar10_train_epoch.npy')

classes = tuple(os.listdir('/local/scratch/jrs596/dat/Forestry_ArableImages_GoogleBing_Licenced_VAE_filtered_unsplit'))

#classes = tuple(os.listdir('/local/scratch/jrs596/dat/test2/images'))

# In[ ]:

# ## Use t-SNE for dimension reduction

# ### compressed to 2-dimension

# first reduce dimensionality before feeding to t-sne
print('Running PCA')
pca = PCA(n_components=100)
X_pca = pca.fit_transform(z_train) 

exp_var_pca = pca.explained_variance_ratio_
#
# Cumulative sum of eigenvalues; This will be used to create step plot
# for visualizing the variance explained by each principal component.
#
cum_sum_eigenvalues = np.cumsum(exp_var_pca)
#
# Create the visualization plot
#
plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#plt.plot(np.cumsum(pca.explained_variance_ratio_))
#plt.xlabel('number of components')
#plt.ylabel('cumulative explained variance')
#plt.show()

# In[ ]:
print('Running t-NSE')
tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=10000, learning_rate='auto')
z_embed = tsne.fit_transform(X_pca)


# In[ ]:


fig = plt.figure(figsize=(12, 10))
plots = []
markers = ['o', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']  # select different markers
for i, c in enumerate(classes):
    ind = (y_train == i).tolist()
    color = cm.jet([i / len(classes)] * sum(ind))
    # plot each category one at a time 
    plots.append(plt.scatter(z_embed[ind, 0], z_embed[ind, 1], c=color, s=8, label=i))

plt.axis('off')
plt.xlim(-150, 150)
plt.ylim(-150, 150)
plt.legend(plots, classes, fontsize=14, loc='upper right')
plt.title('{} (t-SNE: -dim -> 2-dim)', fontsize=14)
plt.savefig("./ResNetVAE__embedded_plot.png", bbox_inches='tight', dpi=600)
#plt.show()

