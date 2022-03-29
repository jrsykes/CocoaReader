

pth = '/local/scratch/jrs596/dat/Forestry_ArableImages_GoogleBing_Licenced_clean_unorganised/test/Fraxinus_Healthy216.jpg'
#pth = '/home/jamiesykes/Documents/MRCNN_annotations_and_data_test/task_test-2021_11_19_14_02_28-coco 1.0/images/1644850391.8956632'
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread(pth)
imgplot = plt.imshow(img)
plt.show()