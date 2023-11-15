import os
import shutil

# Define the paths for train, val, and test directories
val_dir = '/local/scratch/jrs596/dat/FAIGB/FAIGB_FinalSplit_700_backup/val'
test_dir = '/local/scratch/jrs596/dat/FAIGB/FAIGB_FinalSplit_700_backup/test'


def combine_test_val_images(test_dir, val_dir):
    # Check if test and val directories exist
    if not os.path.exists(test_dir):
        print(f"Error: {test_dir} directory does not exist!")
        return
    if not os.path.exists(val_dir):
        print(f"Error: {val_dir} directory does not exist!")
        return

    # Loop through the class directories in the test directory
    for class_name in os.listdir(test_dir):
        test_class_dir = os.path.join(test_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)

        # Check if the class directory exists in val
        if not os.path.exists(val_class_dir):
            os.makedirs(val_class_dir)

        # Loop through the images in the class directory of test
        for image_name in os.listdir(test_class_dir):
            test_image_path = os.path.join(test_class_dir, image_name)
            val_image_path = os.path.join(val_class_dir, image_name)

            # Copy the image from test to val directory
            shutil.copy(test_image_path, val_image_path)
            print(f"Copied {image_name} from {test_class_dir} to {val_class_dir}")

combine_test_val_images(test_dir, val_dir)