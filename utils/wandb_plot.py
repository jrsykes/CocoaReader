#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/local/scratch/jrs596/dat/wandb_export_2023-01-23T08_01_59.684+00_00.csv')

f1 = df['Best_F1']
input_size = df['input_size']
BN_momentum = df['batchnorm_momentum']

for index, row in df.iterrows():
    if row['Best_F1'] >= round(max(f1),6):
        max_f1 = row['Best_F1']
        max_f1_index = index
        max_f1_input_size = input_size[max_f1_index]
        max_f1_BN_momentum = BN_momentum[max_f1_index]

# Plot best f1 vs input size

plt.plot(input_size, f1, 'o')
plt.xlabel('Image input size (Pixels)')
plt.ylabel('F1')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.plot(input_size, f1, 'o', color='grey', alpha=0.5)

plt.plot(max_f1_input_size, max_f1, 'o', color='black')
plt.annotate('Input size = ' + str(int(max_f1_input_size)), xy=(max_f1_input_size, max_f1), xytext=(max_f1_input_size, max_f1+0.01))
plt.rcParams["figure.figsize"] = (8,8)

plt.rc('xtick', labelsize=18) 
plt.rc('ytick', labelsize=18)

plt.show()
plt.clf()


# Plot best f1 vs batchnorm momentum

plt.plot(BN_momentum, f1, 'o')
plt.xlabel('Batchnorm momentum')
plt.ylabel('F1')
plt.xscale('log')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.plot(BN_momentum, f1, 'o', color='grey', alpha=0.5)

plt.plot(max_f1_BN_momentum, max_f1, 'o', color='black')
plt.annotate('BN mom. = ' + str(max_f1_BN_momentum), xy=(max_f1_BN_momentum, max_f1), xytext=(max_f1_BN_momentum, max_f1+0.01))
plt.rcParams["figure.figsize"] = (8,8)

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

plt.rc('font', **font)

plt.show()
plt.clf()

# %%
