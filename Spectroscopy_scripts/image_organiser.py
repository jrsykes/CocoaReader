#%%
import os
import shutil
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

root = "/home/jamiesykes/Downloads/Double"
dumby_imgs = ["left1.png", "1.JPG", "20230123_161149.jpg", "PXL_20230122_115658050.jpg", "PXL_20230203_115351448.jpg", "PXL_20230122_093124778.jpg"]
print('hello')
#%%Unzip google pixel images

for d in os.listdir(os.path.join(root, "Clean_Ecuador_images")):
    if d != "Demo_files" and d != "ReadMe.txt":
        for f in os.listdir(os.path.join(root, "Clean_Ecuador_images", d)):
            img_dir = os.path.join(root, "Clean_Ecuador_images", d, f)
            image_dirs = os.listdir(img_dir)
            for z in image_dirs:
                if z.endswith("-001.zip"):
                    #unzip to list of files in place                 
                    os.system("unzip -o " + os.path.join(img_dir, z) + " -d " + os.path.join(img_dir, z[:-8]))
                    os.remove(os.path.join(img_dir, z))
                    try:
                    #move os.path.join(img_dir, z[:-8] ,z[:-8]) to parent directory
                        os.rename(os.path.join(img_dir, z[:-8], z[:-8]), os.path.join(img_dir, z[:-8] + "_"))
                    #remove os.path.join(img_dir, z[:-8])
                        os.rmdir(os.path.join(img_dir, z[:-8]))
                        os.rename(os.path.join(img_dir, z[:-8] + "_"), os.path.join(img_dir, z[:-8]))

                    except:
                        pass
                elif z.endswith(".zip"):
                    os.system("unzip -o " + os.path.join(img_dir, z) + " -d " + os.path.join(img_dir, z[:-4]))
                    os.remove(os.path.join(img_dir, z)) 
                    try:
                        os.rename(os.path.join(img_dir, z[:-4], z[:-4]), os.path.join(img_dir, z[:-4] + "_"))
                        os.rmdir(os.path.join(img_dir, z[:-4]))
                        os.rename(os.path.join(img_dir, z[:-4] + "_"), os.path.join(img_dir, z[:-4]))
                    except:
                        pass

    
      
print('Done')

#%% Organise into Early and Late directories

count = 0
other = 0
for d in os.listdir(os.path.join(root, "All_Ecuador_images")):
    if d != "Demo_files" and d != "ReadMe.txt":
        for f in os.listdir(os.path.join(root, "All_Ecuador_images", d)):
            img_dir = os.path.join(root, "All_Ecuador_images", d, f)
            image_dirs = os.listdir(img_dir)
            for z in image_dirs:
                dir = os.path.join(root, "All_Ecuador_images", d, f, z)
                images = os.listdir(dir)
                if z.endswith("Temprano"):
                    #dest = "/home/jamiesykes/Documents/Ecuador_data/EcuadorImages_EL_FullRes/Early"
                    dest = os.path.join(root, "Missing/Early")
                elif z.endswith("Unsure"):
                    #dest = "/home/jamiesykes/Documents/Ecuador_data/EcuadorImages_EL_FullRes/Unsure"
                    dest = os.path.join(root, "Missing/Unsure")
                elif z.endswith("Tarde"):
                    #dest = "/home/jamiesykes/Documents/Ecuador_data/EcuadorImages_EL_FullRes/Late"
                    dest = os.path.join(root, "Missing/Late")
                elif z == "Sana":
#                    dest = "/home/jamiesykes/Documents/Ecuador_data/EcuadorImages_EL_FullRes/Late"
                    dest = os.path.join(root, "Missing/Late")
                elif z.startswith("Enfermedad"):
       #             dest = "/home/jamiesykes/Documents/Ecuador_data/EcuadorImages_EL_FullRes/Late"
                    dest = os.path.join(root, "Missing/Late")
                else:
                    other += len(images)
                    #print full path of image z
                    print("End incorrect")
                    print(os.path.join(root, "All_Ecuador_images", d, f, z))
                    print()
                
                if z.startswith("Monilia"):
                    class_ = "FPR"
                elif z.startswith("Fitoptora"):
                    class_ = "BPR"
                elif z.startswith("Escoba"):
                    class_ = "WBD"
                elif z.startswith("Sana"):
                    class_ = "Healthy"
                elif z.startswith("Virus"):
                    class_ = "Virus"
                elif z.startswith("Enfermedad"):
                    class_ = "Unknown_disease"
                else:
                    print("Start incorrect")
                    print(os.path.join(root, "All_Ecuador_images", d, f, z))
                    print()

                os.makedirs(os.path.join(dest, class_), exist_ok = True)
                for i in images:
                    if i not in dumby_imgs:
                        count += 1
                        #make symbolic link
                        #os.symlink(os.path.join(dir, i), os.path.join(dest, class_, str(time.time()) + i))
                        shutil.copy(os.path.join(dir, i), os.path.join(dest, class_, str(time.time()) + i))
                        #open method used to open different extension image file
                        #im = Image.open(os.path.join(dir, i)) 
                        #im1 = im.resize((1200,1200))
                        #im1.save(os.path.join(dest, class_, str(time.time()) + i))
                    elif i in dumby_imgs:
                        pass
                    else:
                        other += 1
                        print(i)
                    
print("Image count: " + str(count))    
print("Filtered image count: " + str(other))   
print("Total image count: " + str(other + count))   

   
#%%


#define dictionary of classes and values
classes = {}

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
            classes[key] = len(os.listdir(os.path.join(dir_, i, j)))

#%%

print(classes)

#%%
#plow classes dict as bar graph
plt.bar(range(len(classes)), list(classes.values()), align='center')
plt.xticks(range(len(classes)), list(classes.keys()))
#rotate x axis labels by 45 degrees
plt.xticks(rotation=45)
#y axis label = "Number of images"
plt.ylabel("Number of images")
#set bar color to white
plt.gca().set_facecolor('white')

#show total of all classes
for i, v in enumerate(list(classes.values())):
    plt.text(i - 0.25, v + 5, str(v), color='black', fontweight='bold')

#show total number of images
plt.text(5, 1400, "Total images: " + str(sum(list(classes.values()))), color='black', fontweight='bold')


#remove top and right borders
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

#set color of classes
plt.gca().get_children()[0].set_color('green')
plt.gca().get_children()[1].set_color('red')
plt.gca().get_children()[2].set_color('red')
plt.gca().get_children()[3].set_color('red')
plt.gca().get_children()[4].set_color('blue')
plt.gca().get_children()[5].set_color('blue')
plt.gca().get_children()[6].set_color('blue')


plt.show()











#%%


combined_classes = {"Healthy": classes["Healthy"], "BPR": classes["BPR_Early"]+classes["BPR_Late"], 
                    "FPR": classes["FPR_Early"]+classes["FPR_Late"], "WBD": classes["WBD_Early"]+classes["WBD_Late"]}

#Web dataset
# combined_classes['Healthy'] += 577
# combined_classes['BPR'] += 187
# combined_classes['FPR'] += 213
# combined_classes['WBD'] += 64

#plow classes dict as bar graph
plt.bar(range(len(combined_classes)), list(combined_classes.values()), align='center')
plt.xticks(range(len(combined_classes)), list(combined_classes.keys()))
#rotate x axis labels by 45 degrees
plt.xticks(rotation=45)
#y axis label = "Number of images"
plt.ylabel("Number of images")
#set bar color to white
plt.gca().set_facecolor('white')

#show total of all classes
for i, v in enumerate(list(combined_classes.values())):
    plt.text(i - 0.25, v + 5, str(v), color='black', fontweight='bold')

#show total number of images
plt.text(2.5, 2000, "Total images: " + str(sum(list(combined_classes.values()))), color='black', fontweight='bold')


#remove top and right borders
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)




plt.show()
# %%

import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Set the directory path
dir_ = "/home/jamiesykes/Downloads/Double/EcuadorWebImages_EasyDif_FinalClean"

# Create an empty dictionary to store Make-Model frequency count for each class
classes = {}

# Create a dictionary to store color code for each Make-Model combination
color_dict = {}


# Loop through each directory in the main directory
for i in os.listdir(dir_):
    if i != "ReadMe.md":
        classes_ = os.listdir(os.path.join(dir_, i))
        for j in classes_:
            for z in os.listdir(os.path.join(dir_, i, j)):
                # Get the Make-Model information from the image metadata
                im = Image.open(os.path.join(dir_, i, j, z))
                exif_data = im.getexif()
                make_model = exif_data.get(271)  # 271 is the tag ID for Make and Model information in EXIF data
                # Create the key for the dictionary with the Make-Model information
                if make_model is not None:
                    key = make_model.split(" ")[0]
                    if key == "SAMSUNG" or key == "samsung":
                        key = "Samsung"
                #     if key not in color_dict:
                #         # Assign a color to the Make-Model combination if it's not already in the color dictionary
                #         color_dict[key] = np.random.rand(3,)
                else:
                #     # If Make-Model information is not available, set the key to "Unknown"
                     key = "Unknown"
                #     if key not in color_dict:
                #         # Assign a random color to "Unknown" if it's not already in the color dictionary
                #         color_dict[key] = np.random.rand(3,)
                # Add the count of images for the Make-Model combination to the classes dictionary
                if key in classes:
                    classes[key] += 1
                else:
                    classes[key] = 1



# Create a list of unique Make-Model combinations
unique_keys = list(classes.keys())
#unique_colors = [color_dict[key] for key in unique_keys]

# Create a bar graph using matplotlib with the x-axis ticks set to the unique Make-Model combinations
plt.bar(range(len(unique_keys)), list(classes.values()), align='center')#, color=unique_colors)
plt.xticks(range(len(unique_keys)), unique_keys)
# Rotate the x-axis labels by 45 degrees for readability
plt.xticks(rotation=45)
# Set the y-axis label to "Number of images"
plt.ylabel("Number of images")
# Set the bar color to white
plt.gca().set_facecolor('white')

# Add text labels to each bar showing the count for that class
for i, v in enumerate(list(classes.values())):
    plt.text(i - 0.25, v + 5, str(v), color='black', fontweight='bold')

# Show total number of images
plt.text(5, 1400, "Total images: " + str(sum(list(classes.values()))), color='black', fontweight='bold')

# Remove top and right borders
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.show()

# %%

print(classes)
# %%
