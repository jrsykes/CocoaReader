from PIL import Image

# Creating a 500x500 pixel image with black color (RGB: 0,0,0)
img_size = (500, 500)
black_color = (0, 0, 0)
img = Image.new("RGB", img_size, black_color)

# Save the image to a file
img.save("/users/jrs596/scratch/dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_Compress500_split/Difficult/val/Healthy/dumby.jpg", "JPEG")
img.save("/users/jrs596/scratch/dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_Compress500_split/Unsure/val/Healthy/dumby.jpg", "JPEG")
img.save("/users/jrs596/scratch/dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_Compress500_split/Difficult/val/NotCocoa/dumby.jpg", "JPEG")
img.save("/users/jrs596/scratch/dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_Compress500_split/Unsure/val/NotCocoa/dumby.jpg", "JPEG")
