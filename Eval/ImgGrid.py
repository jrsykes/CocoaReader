from PIL import Image
import os

root = '/home/userfs/j/jrs596/scripts/CocoaReader'

# Open the images
img1 = Image.open(os.path.join(root, 'gradcam_DisNet-pico-RGB.png'))
img2 = Image.open(os.path.join(root, 'gradcam_EfficientNet-RGB.png'))
img3 = Image.open(os.path.join(root, 'gradcam_ResNet18-RGB.png'))
img4 = Image.open(os.path.join(root, 'gradcam_DisNet-pico2-RGB.png'))

# Assuming the images are the same size, get dimensions of the first image
width, height = img1.size
width2, height2 = img3.size

# Create a new image twice as wide and twice as high as the original
new_img = Image.new('RGBA', (width+width2, height+height2))

# Paste the images into the new image
new_img.paste(img1, (0, 0))
new_img.paste(img2, (width, 0))
new_img.paste(img3, (0, height))
new_img.paste(img4, (width, height))

# Save the new image
new_img.save('GradCamRGB.png')


# Open the images
img1 = Image.open(os.path.join(root, 'gradcam_DisNet-pico-IR.png'))
img2 = Image.open(os.path.join(root, 'gradcam_efficientnet-IR.png'))
img3 = Image.open(os.path.join(root, 'gradcam_resnet18-IR.png'))
img4 = Image.open(os.path.join(root, 'gradcam_disnet-pico2-IR.png'))

# Assuming the images are the same size, get dimensions of the first image
#width, height = img3.size

# Create a new image twice as wide and twice as high as the original
new_img = Image.new('RGBA', (width+width2, height+height2))

# Paste the images into the new image
new_img.paste(img1, (0, 0))
new_img.paste(img2, (width, 0))
new_img.paste(img3, (0, height))
new_img.paste(img4, (width, height))

# Save the new image
new_img.save('GradCamIR.png')
