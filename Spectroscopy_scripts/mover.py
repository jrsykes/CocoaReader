import os
import shutil
import subprocess

class_ = "WBD"

# Define the directories to move the images to
directory_u = "/home/jamiesykes/Downloads/Double/EcuadorImages_ED_FullRes+WebData/Unsure/" + class_
directory_d = "/home/jamiesykes/Downloads/Double/EcuadorImages_ED_FullRes+WebData/Difficult/" + class_
directory_e = "/home/jamiesykes/Downloads/Double/EcuadorImages_ED_FullRes+WebData/Easy/" + class_

# Get the path of the directory containing the images
directory_path = "/home/jamiesykes/Downloads/Double/Missing/Early/" + class_


# Loop through all files in the directory
for filename in os.listdir(directory_path):
    # Check if the file is an image (you can modify this to check for specific file extensions)
    #if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
       # Open the image
    image_path = os.path.join(directory_path, filename)
    subprocess.run(['xdg-open', image_path])
       
    # Ask the user for input
    user_input = input(f"Move {filename} to directory A, B, or C? (type 'a', 'b', or 'c', or any other key to skip): ")
    # Move the image to the appropriate directory based on user input1
    if user_input == "1":
        #move and override
        shutil.move(image_path, os.path.join(directory_e, filename), copy_function=shutil.copy2)
    elif user_input == "2":
        shutil.move(image_path, os.path.join(directory_d, filename), copy_function=shutil.copy2)
    elif user_input == "3":
        shutil.move(image_path, os.path.join(directory_u, filename), copy_function=shutil.copy2)

