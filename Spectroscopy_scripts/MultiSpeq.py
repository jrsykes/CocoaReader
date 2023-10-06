#%%
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import wrap
import numpy as np
import textwrap

root = "/home/jamiesykes/Documents/Ecuador_data/CocoaSpectroscopy/Photosynq-cocoa-disease-detection.csv"

#load csv as pandas df
df = pd.read_csv(root)


#%%   
#removes values that contacin "virus"
data = df[df["Suspected disease state"] != "Unidentified virus"]
data = data[data["Suspected disease state"] != "Unidentified virus adjacent tissue"]
#edit entries for "Witches broom" to be "Witches' broom (affected tissue)"
data["Suspected disease state"] = data["Suspected disease state"].replace("Witches broom", "Witches' broom (affected tissue)")
data["Suspected disease state"] = data["Suspected disease state"].replace("Witches broom adjacent tissue", "Witches' broom (adjacent tissue)")

#%%

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

labels = ['\n'.join(wrap(l, 20)) for l in data["Suspected disease state"].values]

# Calculate mean and standard error for each group
mean = data.groupby('Suspected disease state')['NPQt'].mean()
standard_error = data.groupby('Suspected disease state')['NPQt'].std() / np.sqrt(data.groupby('Suspected disease state').size())

# Create labels for each group
labels = ['\n'.join(wrap(l, 20)) for l in mean.index]

# Plot mean with error bars
ax.errorbar(labels, mean, yerr=standard_error, fmt='o', color='black')

#remove grid lines
ax.grid(False)
#remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#Add Y axis label
ax.set_ylabel('NPQt ± 1 SE', fontsize=12)
#set size of x axis labels
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

plt.show()

# %%

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

labels = ['\n'.join(wrap(l, 20)) for l in data["Suspected disease state"].values]

# Calculate mean and standard error for each group
mean = data.groupby('Suspected disease state')['Phi2'].mean()
standard_error = data.groupby('Suspected disease state')['Phi2'].std() / np.sqrt(data.groupby('Suspected disease state').size())

# Create labels for each group
labels = ['\n'.join(wrap(l, 20)) for l in mean.index]

# Plot mean with error bars
ax.errorbar(labels, mean, yerr=standard_error, fmt='o', color='black')

#remove grid lines
ax.grid(False)
#remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#Add Y axis label
ax.set_ylabel('Phi2 ± 1 SE', fontsize=12)
#set size of x axis labels
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

plt.show()# %%

# %%
import seaborn as sns
from textwrap import wrap
import matplotlib.pyplot as plt

# Desired order and wrapped labels
desired_order = ['Black pod rot', 'Healthy', 'Frosty pod rot', 'Witches\' broom (adjacent tissue)', 'Witches\' broom (affected tissue)']
wrapped_order = ['\n'.join(textwrap.wrap(label, width=17)) for label in desired_order]

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

# Plot violin plot
sns.violinplot(x=data["Suspected disease state"], y=data["NPQt"], ax=ax, inner="box", color="white", order=desired_order)

# Adjust aesthetics
ax.set_xticklabels(wrapped_order, fontsize=14, rotation=45)  # Use wrapped_order for x-axis labels
ax.set_ylabel('NPQt', fontsize=14)
ax.set_xlabel('')
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig('/home/jamiesykes/Documents/Ecuador_data/plots/NPQt_violin.png', dpi=300, bbox_inches='tight')
plt.show()


# %%
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

# Get unique values from 'Suspected disease state' column
unique_states = data["Suspected disease state"].unique()

# Create labels for each unique value
labels = ['\n'.join(wrap(l, 20)) for l in unique_states]



sns.violinplot(x=data["Suspected disease state"], y=data["Phi2"], ax=ax, inner="box", color="white", order=desired_order)

# Adjust aesthetics
ax.set_xticklabels(wrapped_order, fontsize=14, rotation=45)  # Use wrapped_order for x-axis labels
ax.set_ylabel('Phi2', fontsize=14)
ax.set_xlabel('')
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig('/home/jamiesykes/Documents/Ecuador_data/plots/Phi2_violin.png', dpi=300, bbox_inches='tight')

plt.show()

# %%


import matplotlib.pyplot as plt
from textwrap import wrap
import numpy as np

# Function to calculate the margin of error for 95% CI
def margin_of_error(std, n):
    Z = 1.96  # Z-score for 95% CI
    return Z * (std / np.sqrt(n))

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

labels = ['\n'.join(wrap(l, 20)) for l in data["Suspected disease state"].values]

# Calculate mean and margin of error for each group
mean = data.groupby('Suspected disease state')['NPQt'].mean()
mo_error = data.groupby('Suspected disease state')['NPQt'].std().apply(lambda x: margin_of_error(x, len(data)))

# Create labels for each group
labels = ['\n'.join(wrap(l, 20)) for l in mean.index]

# Plot mean with error bars
ax.errorbar(labels, mean, yerr=mo_error, fmt='o', color='black')

# Formatting
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('NPQt (95% CI)', fontsize=12)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

plt.show()

# %%

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

labels = ['\n'.join(wrap(l, 20)) for l in data["Suspected disease state"].values]

# Calculate mean and margin of error for each group
mean = data.groupby('Suspected disease state')['Phi2'].mean()
mo_error = data.groupby('Suspected disease state')['Phi2'].std().apply(lambda x: margin_of_error(x, len(data)))

# Create labels for each group
labels = ['\n'.join(wrap(l, 20)) for l in mean.index]

# Plot mean with error bars
ax.errorbar(labels, mean, yerr=mo_error, fmt='o', color='black')

# Formatting
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('Phi2 (95% CI)', fontsize=12)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

plt.show()
