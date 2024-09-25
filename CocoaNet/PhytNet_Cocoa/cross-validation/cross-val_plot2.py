#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
# Step 1: Organize the Data into Two DataFrames

root = '/local/scratch/jrs596/dat/cross-val_data/'
# Load the CSV files into pandas DataFrames
train_model1 = pd.read_csv(os.path.join(root, 'DisNet_CrossVal_IR_val_metrics.csv'), header=None)
train_model2 = pd.read_csv(os.path.join(root, 'ResNet18_CrossVal_IR_train_metrics.csv'), header=None)
train_model3 = pd.read_csv(os.path.join(root, 'ResNet18_CrossVal_RGB_train_metrics.csv'), header=None)

test_model1 = pd.read_csv(os.path.join(root,'DisNet_CrossVal_IR_val_metrics.csv'), header=None)
test_model2 = pd.read_csv(os.path.join(root,'ResNet18_CrossVal_IR_val_metrics.csv'), header=None)
test_model3 = pd.read_csv(os.path.join(root,'ResNet18_CrossVal_RGB_val_metrics.csv'), header=None)

#%%

train_model4 = pd.read_csv(os.path.join(root, 'ConvNext_tiny_CrossVal_IR_train_metrics.csv'), header=None)
train_model4 = train_model4.T
train_model4.columns = train_model4.iloc[0]
train_model4 = train_model4.drop(train_model4.index[0])

# test_model4 = pd.read_csv(os.path.join(root, 'ConvNext_tiny_CrossVal_IR_val_metrics.csv'), header=None)
# test_model4 = test_model4.T
# test_model4.columns = test_model4.iloc[0]
# test_model4 = test_model4.drop(test_model4.index[0])

# print(train_model4['f1'].mean())
# print(test_model4.iloc[:,0].mean())


#%%
# Organize the data
dataframes = []

models_train = [train_model1, train_model2, train_model3]
models_test = [test_model1, test_model2, test_model3]
model_names = ["PhtyNet IR", "ResNet18 IR", "ResNet18 RGB"]

for i, (train_df, test_df) in enumerate(zip(models_train, models_test)):
    for df, phase in zip([train_df, test_df], ["train", "val"]):
        df_melted = df.melt(id_vars=[0], var_name="fold", value_name="value")
        df_melted["model"] = model_names[i]
        df_melted["phase"] = phase
        dataframes.append(df_melted)

# Combine all dataframes
df_combined = pd.concat(dataframes, ignore_index=True)
df_combined = df_combined[df_combined[0] != 'loss']
df_combined = df_combined[df_combined[0] != 'acc']


#%%
x_axis_labels = ["F1", "Precision", "Recall", "BPR F1", "FPR F1", "Healthy F1", "WBD F1"]
# Define font sizes
tick_fontsize = 18
legend_fontsize = 18
legend_title_fontsize = 18

# Plot the data
for phase in ["train", "val"]:
    plt.figure(figsize=(15, 8))
    palette = ["#3c6997", "#d62839", "#718355"]
    sns.set_palette(palette)
    
    ax = sns.violinplot(data=df_combined[df_combined["phase"] == phase], 
                   x=0, y="value", hue="model", dodge=True)  # Removed split=True and added dodge=True


    ax.set_xlabel('Benchmark metric', fontsize=tick_fontsize)  # Remove x-axis label
    ax.set_ylabel("Probability density of metric", fontsize=tick_fontsize)  # Set y-axis label
    ax.set_xticklabels(x_axis_labels, fontsize=tick_fontsize)  # Set new x-axis class labels with font size
    ax.tick_params(axis='y', labelsize=tick_fontsize)  # Adjust y-axis tick label font size
    ax.set_ylim(0, 1.2)

    # Set the title and adjust the legend position based on the phase
    if phase == "train":
        # ax.set_title("Training", fontsize=20)
        plt.legend(title="Model", loc="upper left", prop={'size': legend_fontsize}, 
                   title_fontsize=legend_title_fontsize, bbox_to_anchor=(0, 1.05))
    else:
        # ax.set_title("Validation", fontsize=20)
        ax.legend().set_visible(False)  # Hide the legend for "val" phase
                  
    sns.despine()  # This will remove the top and right spines
    plt.savefig(os.path.join(root, f"cross-val_{phase}.png"), dpi=300, bbox_inches='tight')

    plt.show()


# %%
