#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 20:08:59 2020

@author: jamie
"""
from datetime import datetime

import tensorflow as tf
import keras
import numpy as np
import os

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

batch_size = 32
img_height = 180
img_width = 180
epochs = 100

base_dir = '/home/jamie/Documents/cacao_net/disease_dataset (another copy)/'
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
#Prep transfer model layers

from tensorflow.keras import layers
from tensorflow.keras import Model
import wget

url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
# wget.download(url, out = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

    
from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape = (img_height, img_width, 3), 
                                include_top = False, 
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
  layer.trainable = False
  
pre_trained_model.summary()

#%% 



train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  base_dir,
  validation_split=0.1,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  base_dir,
  validation_split=0.1,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)
#%%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break
#%% 
#configure data set

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
#%%


class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accuracy')>0.99):
                print("\n Reached 99% accuracy so cancelling training!")
                self.model.stop_training = True
callbacks=myCallback()



#%%

n_classes = len(class_names)

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

from tensorflow.keras.optimizers import RMSprop

x = layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(470, activation='relu')(x)
x = layers.Dense(870, activation='relu')(x)
x = layers.Dense(470, activation='relu')(x)
# x = layers.Dense(470, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)                  
# Add a final sigmoid layer for classification
x = layers.Dense  (n_classes, activation='softmax')(x)           

model = Model(pre_trained_model.input, x) 

model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'SparseCategoricalCrossentropy', 
              metrics = ['accuracy'])

#%%
start = datetime.now()                                
# Train the Model
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)
#, callbacks=[callbacks])

end = datetime.now()
runtime= end-start
print ("Runtime: " + str(runtime))


model.save('/home/jamie/Documents/cacao_net/cacao_net_model')
# model.summary()
#%%
# Plot the chart for accuracy and loss on both training and validation
#get_ipython().run_line_magic('matplotlib', 'inline')
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
#Predict

# img_path = '/home/jamie/Downloads/Sporulating-lesions-of-frosty-pod-rot-of-cocoa-caused-by-M-roreri-Costa-Rica-C-J_Q320.jpg'
# img_path = '/home/jamie/Downloads/c0168421-400px-wm.jpg'
# img_path = '/home/jamie/Documents/cacao_net/disease_dataset (another copy)/fpr/55805a0f34c29.jpg'
img_path = '/home/jamie/Downloads/download(1).jpeg'

# model = tf.keras.models.load_model('/home/jamie/Documents/cacao_net/cacao_net_model')

img = keras.preprocessing.image.load_img(
    img_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)


import matplotlib.pyplot as plt
plt.imshow(img)

#%%

#Convert saved model to tflight

#Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('/home/jamie/Documents/cacao_net/cacao_net_model') # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  os.chdir('/home/jamie/Documents/cacao_net/cacao_net_lite_model')
  f.write(tflite_model)

#%%
#Load and predict with tflite model

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path='/home/jamie/Documents/cacao_net/cacao_net_lite_model/model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
# input_shape = input_details[0]['shape']
# input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
input_data = img_array
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(class_names[np.argmax(output_data)])