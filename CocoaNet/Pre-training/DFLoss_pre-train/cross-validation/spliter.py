import os
import numpy as np
import PIL.Image as Image

# Root directory where the classes subdirectories are located
root_dir = '/users/jrs596/scratch/dat/IR_RGB_Comp_data/compiled_IR'
# Destination directory where to create the 10 folders for the folds
dest_dir = '/users/jrs596/scratch/dat/IR_RGB_Comp_data/cross-val_IR'

# Get the classes directories
class_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

# Create 10 directories for the folds
for i in range(10):
    os.makedirs(os.path.join(dest_dir, f'fold_{i}', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, f'fold_{i}', 'val'), exist_ok=True)

# Go through each class directory and create symbolic links in the folds directories
for class_dir in class_dirs:
    class_name = os.path.basename(class_dir)
    images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    np.random.shuffle(images)  # Randomly shuffle the images
    images_count = len(images)

    for i in range(10):
        # Determine the start and end indices for the images for validation set for this fold
        start_index = i * images_count // 10
        end_index = (i + 1) * images_count // 10 if i != 9 else images_count  # Ensure the last fold includes all remaining images

        # Create symbolic links in the train and validation subdirectories
        for j, img in enumerate(images):
            if start_index <= j < end_index:
                subdir = 'val'
            else:
                subdir = 'train'

            link_dir = os.path.join(dest_dir, f'fold_{i}', subdir, class_name)
            os.makedirs(link_dir, exist_ok=True)
            
            link_name = os.path.join(link_dir, os.path.basename(img))
            if not os.path.exists(link_name):  # Avoid duplicating links
                #os.symlink(img, link_name)
                #open image with PIL, compress and save
                im = Image.open(img)
                im_ = im.resize((1000,1000))
                im_.save(link_name)
