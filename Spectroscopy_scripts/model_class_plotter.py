#%%
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#%%
#define empty dataframe with columns for class, make-model, and frequency
df = pd.DataFrame(columns=['Class', 'Make-Model', 'Frequency'])

dir_ = "/home/jamiesykes/Downloads/Double/EcuadorWebImages_EasyDif_FinalClean"
for i in os.listdir(dir_):
    if i != "ReadMe.md":
        classes_ = os.listdir(os.path.join(dir_, i))
        #print(i + ": ")
        for j in classes_:
            #print(j + ": " + str(len(os.listdir(os.path.join(dir_, i, j)))))
            if j == "Healthy":
                key = "Healthy"
            else:
                key = j + "_" + i
            #classes[key] = len(os.listdir(os.path.join(dir_, i, j)))
            models = {}
            for img in os.listdir(os.path.join(dir_, i, j)):
                # Get the Make-Model information from the image metadata
                im = Image.open(os.path.join(dir_, i, j, img))
                exif_data = im.getexif()
                make_model = exif_data.get(271)  # 271 is the tag ID for Make and Model information in EXIF data
                            
                if make_model is not None:
                    key = make_model.split(" ")[0]
                    if key == "SAMSUNG" or key == "samsung":
                        key = "Samsung"
                else:
                #     # If Make-Model information is not available, set the key to "Unknown"
                     key = "Unknown"
                if key in models:
                    models[key] += 1
                else:
                    models[key] = 1
            
            for key in models.keys():
                out = [j, key, models[key]]
                #append out to df as row
                df.loc[len(df)] = out

print(df)

#%%



# %%
import matplotlib.pyplot as plt

#rename any Make-Model values == Unknow to Other
df.loc[df['Make-Model'] == "Unknown", 'Make-Model'] = "Other"
df.loc[df['Make-Model'] == "vivo", 'Make-Model'] = "Vivo"

#for row in df_agg: if Frequencey < 10, set Make_Model to "Other"
df.loc[df['Frequency'] < 10, 'Make-Model'] = "Other"
# Aggregate duplicate rows
df_agg = df.groupby(['Class', 'Make-Model']).sum().reset_index()

df_pivot = df_agg.pivot(index='Class', columns='Make-Model', values='Frequency')
df_pivot.plot(kind='bar', stacked=True, figsize=(6, 5), width=0.6)
plt.ylabel('n images')
#remove grid lines
plt.grid(False)
#move legend outside of plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#remove top and right borders
sns.despine()
#remove x-axis label
plt.xlabel('')
#set bar lables to 90 degrees
plt.xticks(rotation=45)


plt.show()




#%%

import numpy as np

# Aggregate duplicate rows
df_agg = df.groupby(['Class', 'Make-Model']).sum().reset_index()

df_pivot = df_agg.pivot(index='Make-Model', columns='Class', values='Frequency')
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df_pivot, cmap='YlGnBu', annot=True, fmt='g', ax=ax)
ax.set_xlabel('Class')
ax.set_ylabel('Make-Model')
ax.set_title('Frequency of Make-Model by Class')

#%%