#%%
from PIL import Image
import os

# Define the directories where your training and validation images are stored
data_directories = {
    'train': 'scratch/dat/FAIGB/FAIGB_700_30-10-23_split/train',
    'val': 'scratch/dat/FAIGB/FAIGB_700_30-10-23_split/val'
}

# Function to convert images
def convert_images(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.gif')):  # Add or remove file types as needed
                file_path = os.path.join(subdir, file)
                with Image.open(file_path) as img:
                    # Check if the image has a 'P' mode which indicates it's a palette-based image
                    if img.mode == 'P':
                        # Check if the image has transparency information
                        if 'transparency' in img.info:
                            print(f"Converting image {file_path} to RGBA.")
                            # Convert the image to RGBA
                            img = img.convert('RGBA')
                            # Save the image back to the same path or a new one
                            img.save(file_path)

# Convert images in both train and val directories
for dataset_type, directory in data_directories.items():
    print(f"Processing {dataset_type} data...")
    convert_images(directory)

print("Conversion complete.")


# %%
