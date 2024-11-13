import torch
import clip
from PIL import Image
import os
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# Load CLIP model and preprocessing
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load taxonomy CSV file
taxonomy_file = "scratch/dat/flowers102_split/flowers_taxonomy.csv"
taxonomy_df = pd.read_csv(taxonomy_file)

# Specify the directory containing images
image_dir = "scratch/dat/flowers102_split/val/"

# Create a list to store embeddings and labels (taxonomic levels)
embeddings = []
labels = []

# Process each image and get embeddings
for index, row in taxonomy_df.iterrows():
    label = row['order']  # Change this to 'family' or another level if desired
    species_name = row['label']
    for image in len(os.listdir(os.path.join(image_dir, species_name))):
        img_path = os.path.join(image_dir, species_name, os.listdir(os.path.join(image_dir, species_name))[image])  

        try:
            # Preprocess image
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

            # Get CLIP embedding
            with torch.no_grad():
                image_features = model.encode_image(image).cpu().numpy()

            embeddings.append(image_features)
            labels.append(label)

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

# Convert embeddings and labels to numpy arrays
embeddings = np.vstack(embeddings)
labels = np.array(labels)

# Reduce dimensions with UMAP
umap_model = umap.UMAP(n_components=3, random_state=42)
reduced_embeddings = umap_model.fit_transform(embeddings)

# Plot 3D point cloud colored by taxonomic level
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Use a color map for different labels
unique_labels = np.unique(labels)
colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))
color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

for label in unique_labels:
    idx = np.where(labels == label)
    ax.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], reduced_embeddings[idx, 2],
               color=color_map[label], label=label)

ax.set_title("3D UMAP Projection of CLIP Embeddings")
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_zlabel('UMAP 3')

# Add a legend
ax.legend(loc='best')

# Show plot
plt.show()
# Save plot
plt.savefig("scratch/dat/3d_umap_plot.png")
