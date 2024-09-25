#%%
import os
import shutil
import scipy.io
from sklearn.model_selection import train_test_split

# Load the .mat file
mat_file_path = '/users/jrs596/scratch/dat/imagelabels.mat'
mat_data = scipy.io.loadmat(mat_file_path)
labels = mat_data.get('labels')[0]

# Define paths
original_images_path = '/users/jrs596/scratch/dat/flowers102'
organized_dataset_path = '/users/jrs596/scratch/dat/flowers102_split'

# Create directories for each label
unique_labels = set(labels)
for label in unique_labels:
    train_dir = os.path.join(organized_dataset_path, 'train', str(label))
    test_dir = os.path.join(organized_dataset_path, 'test', str(label))
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

# Create a list of image paths and their corresponding labels
image_paths = []
image_labels = []
for i, label in enumerate(labels):
    image_path = os.path.join(original_images_path, f'image_{i + 1:05d}.jpg')  # Adjusted naming convention
    if os.path.exists(image_path):
        image_paths.append(image_path)
        image_labels.append(label)
    else:
        print(f"Warning: Image {image_path} does not exist.")

# Check if any images were found
if len(image_paths) == 0:
    raise ValueError("No images found. Please check the image directory and naming convention.")

# Split the data into train and test sets
try:
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, image_labels, test_size=0.10, stratify=image_labels, random_state=42)
except ValueError as e:
    print(f"Error during train-test split: {e}")
    raise

# Function to copy images to their respective directories
def copy_images(image_paths, labels, dataset_type):
    for image_path, label in zip(image_paths, labels):
        destination_dir = os.path.join(organized_dataset_path, dataset_type, str(label))
        shutil.copy(image_path, destination_dir)

# Copy images to train and test directories
copy_images(train_paths, train_labels, 'train')
copy_images(test_paths, test_labels, 'test')

print('Images have been organized and split into train and test directories.')# %%

# %%
import torchvision

data = torchvision.datasets.Flowers102(root="/users/jrs596/scratch/dat", split='val', download=True)