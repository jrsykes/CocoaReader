#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 20:08:59 2020

@author: jamie
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#%% 
# Split training and test data
# import random
# import shutil

# split = .8
# species = os.listdir(base_dir)

# for item in species:
#     os.makedirs(base_dir + '/train/' + item)
#     os.makedirs(base_dir + '/test/' + item)
    
#     files = os.listdir(base_dir + item)
#     random.shuffle(files)
#     n_pics = len(files)
#     train = files[:int(n_pics*split)]
#     test = files[int(n_pics*split):]
            
#     for i in train:
#         shutil.move(base_dir + item + '/' + i, base_dir + '/train/' + item)
#     for i in test:
#         shutil.move(base_dir + item + '/' + i, base_dir + '/test/' + item)

  
#%% 

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


base_dir = '/home/jamie/Documents/weed_net/weed_net_data/'

#%%

from tensorflow.keras import layers
from tensorflow.keras import Model
#!wget --no-check-certificate https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
 #   -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
  
from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
  layer.trainable = False
  
pre_trained_model.summary()

#%% 
#from tensorflow.keras.preprocessing.image import ImageDataGenerator

# import keras_preprocessing
# from keras_preprocessing import image
# from keras_preprocessing.image import ImageDataGenerator

# TRAINING_DIR = base_dir + 'train'
# training_datagen = ImageDataGenerator(
#       rescale = 1./255,
# 	    #rotation_range=40,
#       width_shift_range=0.2,
#       height_shift_range=0.2,
#       shear_range=0.2,
#       zoom_range=0.2,
#       horizontal_flip=True,
#       fill_mode='nearest')

# VALIDATION_DIR = base_dir + 'test'
# validation_datagen = ImageDataGenerator(rescale = 1./255)

# train_generator = training_datagen.flow_from_directory(
# 	TRAINING_DIR,
# 	target_size=(150,150),
# 	class_mode='categorical',
#   batch_size=100
# )

# validation_generator = validation_datagen.flow_from_directory(
# 	VALIDATION_DIR,
# 	target_size=(150,150),
# 	class_mode='categorical',
#   batch_size=100)


#%% 
data_dir = '/home/jamie/Documents/weed_net/weed_net_data_copy/'

batch_size = 32
img_height = 150
img_width = 150


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)


#%%
#Augmentation

data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
    layers.experimental.preprocessing.Rescaling(1./255, offset=0.0, name=None),

  ]
)

#%%



class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accuracy')>0.99):
                print("\n Reached 99% accuracy so cancelling training!")
                self.model.stop_training = True
callbacks=myCallback()
#%%

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

from tensorflow.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(335, activation='relu')(x)
x = layers.Dense(670, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)                  
# Add a final sigmoid layer for classification
x = layers.Dense  (67, activation='softmax')(x)           

model = Model( pre_trained_model.input, x) 

model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])
#%%
                                    
# Train the Model
history = model.fit_generator(train_ds, epochs=2, validation_data=val_ds, callbacks=[callbacks])

#model.save('/home/jamie/Documents/weed_net/weed_net_model')


# In[6]:


# Plot the chart for accuracy and loss on both training and validation
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
acc = history.history['accuracy']# Your Code Here
val_acc = history.history['val_accuracy']# Your Code Here
loss = history.history['loss']# Your Code Here
val_loss = history.history['val_loss']# Your Code Here

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#%%
from tensorflow import keras



img_path = '/home/jamie/Documents/weed_net/weed_net_data/test/aethusa_cynapium/10110.jpg'

img = keras.preprocessing.image.load_img(
    img_path, target_size=(150, 150)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

