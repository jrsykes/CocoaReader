#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/local/scratch/jrs596/dat/wandb_export_2023-01-30T16_47_11.784+00_00.csv')


f1 = df['Best_F1']
input_size = df['input_size']
BN_momentum = df['batchnorm_momentum']


for index, row in df.iterrows():
    #print(row['Best_F1'], round(max(f1),6))
    if row['Best_F1'] >= max(f1):
        max_f1 = row['Best_F1']
        max_f1_index = index
        max_f1_input_size = input_size[max_f1_index]
        max_f1_BN_momentum = BN_momentum[max_f1_index]



fontsize = 22
# Plot best f1 vs input size
plt.subplot(2, 1, 2)
#set gap between plots
plt.subplots_adjust(hspace=0.3)
#Lable two plats as A and B
plt.text(0.5, 0.47, 'A', fontsize=fontsize, weight='bold', transform=plt.gcf().transFigure)
plt.text(0.5, 0.03, 'B', fontsize=fontsize, weight='bold', transform=plt.gcf().transFigure)


plt.rc('xtick', labelsize=fontsize) 
plt.rc('ytick', labelsize=fontsize)

plt.plot(input_size, f1, 'o')
plt.xlabel('Image input size (Pixels)', fontsize=fontsize)
plt.ylabel('F1', fontsize=fontsize)
ax = plt.gca()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


plt.plot(input_size, f1, 'o', color='grey', alpha=0.5)

plt.plot(max_f1_input_size, max_f1, 'o', color='black')
plt.annotate('Input size = ' + str(int(max_f1_input_size)), xy=(max_f1_input_size, max_f1), xytext=(max_f1_input_size, max_f1+0.01), fontsize=fontsize)





# Plot best f1 vs batchnorm momentum
plt.subplot(2, 1, 1)

plt.plot(BN_momentum, f1, 'o')
plt.xlabel('Batchnorm momentum', fontsize=fontsize)
plt.ylabel('F1', fontsize=fontsize)
plt.xscale('log')
ax = plt.gca()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig = plt.gcf()
fig.set_size_inches(10, 16)

plt.plot(BN_momentum, f1, 'o', color='grey', alpha=0.5)

plt.plot(max_f1_BN_momentum, max_f1, 'o', color='black')
plt.annotate('BN mom. = ' + str(max_f1_BN_momentum), xy=(max_f1_BN_momentum, max_f1), xytext=(max_f1_BN_momentum, max_f1+0.01), fontsize=fontsize)

plt.rc('xtick', labelsize=fontsize) 
plt.rc('ytick', labelsize=fontsize)

# plt.show()
# plt.clf()



plt.show()
#plt.savefig('/local/scratch/jrs596/dat/wandb_plot.png')


# %%
