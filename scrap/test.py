from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import numpy as np
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import os.path

#test_data_root = os.environ['/home/userfs/j/jrs596/scripts/CocoaReader/DALI_extra']
#db_folder = os.path.join(test_data_root, 'db', 'lmdb')
#db_folder = '/local/scratch/jrs596/dat/DisNet_lmdb/test.lmdb'
db_folder = '/home/userfs/j/jrs596/scripts/CocoaReader/DALI_extra/db/lmdb'

jpegs, labels = fn.readers.caffe(path=db_folder)
images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
output = fn.crop_mirror_normalize(
    images,
    dtype=types.FLOAT,
    crop=(224, 224),
    mean=[0., 0., 0.],
    std=[1., 1., 1.],
    crop_pos_x=fn.random.uniform(range=(0, 1)),
    crop_pos_y=fn.random.uniform(range=(0, 1)))


batch_size = 9#

#pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0)
#with pipe:
#    jpegs, labels = fn.readers.caffe(path=db_folder)
#    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
#    output = fn.crop_mirror_normalize(
#        images,
#        dtype=types.FLOAT,
#        crop=(224, 224),
#        mean=[0., 0., 0.],
#        std=[1., 1., 1.],
#        crop_pos_x=fn.random.uniform(range=(0, 1)),
#        crop_pos_y=fn.random.uniform(range=(0, 1)))
#    pipe.set_outputs(output, labels)

#pipe.build()

#pipe_out = pipe.run()


import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


def show_images(image_batch):
    columns = 3
    rows = (batch_size + 1) // (columns)
    fig = plt.figure(figsize = (20,(20 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows*columns):
        plt.subplot(gs[j])
        plt.axis("off")
        img_chw = image_batch.at(j)
        img_hwc = np.transpose(img_chw, (1,2,0))/255.0
        plt.imshow(img_hwc)


#images, labels = pipe_out
show_images(output)