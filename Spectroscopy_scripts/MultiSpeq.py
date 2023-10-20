#%%
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import wrap
import textwrap
import seaborn as sns


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

# Desired order and wrapped labels
desired_order = ['Black pod rot', 'Healthy', 'Frosty pod rot', 'Witches\' broom (adjacent tissue)', 'Witches\' broom (affected tissue)']
wrapped_order = ['\n'.join(textwrap.wrap(label, width=17)) for label in desired_order]

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

# Plot violin plot
# sns.violinplot(x=data["Suspected disease state"], y=data["NPQt"], ax=ax, inner="box", color="white", order=desired_order)
sns.boxplot(x=data["Suspected disease state"], y=data["NPQt"], 
            ax=ax, color="white", order=desired_order, showfliers=False)
sns.stripplot(x=data["Suspected disease state"], y=data["NPQt"], 
              ax=ax, color="white", edgecolor="black", linewidth=0.9, jitter=True, 
              size=6, order=desired_order)


# Adjust aesthetics
ax.text(0.05, 0.95, 'A', transform=ax.transAxes, fontsize=16, verticalalignment='top', 
        horizontalalignment='left')

ax.set_xticklabels(wrapped_order, fontsize=14, rotation=45)  # Use wrapped_order for x-axis labels
ax.set_ylabel('NPQt', fontsize=14)
ax.set_xlabel('')
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig('/home/jamiesykes/Documents/Ecuador_data/plots/NPQt_box.png', dpi=300, bbox_inches='tight')
plt.show()


# %%
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

# Get unique values from 'Suspected disease state' column
unique_states = data["Suspected disease state"].unique()

# Create labels for each unique value
labels = ['\n'.join(wrap(l, 20)) for l in unique_states]



sns.boxplot(x=data["Suspected disease state"], y=data["Phi2"], 
            ax=ax, color="white", order=desired_order, showfliers=False)
sns.stripplot(x=data["Suspected disease state"], y=data["Phi2"], 
              ax=ax, color="white", edgecolor="black", linewidth=0.9, jitter=True, 
              size=6, order=desired_order)
# Adjust aesthetics
ax.text(0.05, 1.1, 'B', transform=ax.transAxes, fontsize=16, verticalalignment='top', horizontalalignment='left')


ax.set_xticklabels(wrapped_order, fontsize=14, rotation=45)  # Use wrapped_order for x-axis labels
ax.set_ylabel('Phi2', fontsize=14)
ax.set_xlabel('Disease state', fontsize=14)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig('/home/jamiesykes/Documents/Ecuador_data/plots/Phi2_box.png', dpi=300, bbox_inches='tight')

plt.show()
