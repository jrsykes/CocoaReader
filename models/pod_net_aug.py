#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 29 13:11:17 2021

@author: jamie
"""

from datetime import datetime

import tensorflow as tf
import keras
import numpy as np
import os

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


img_height = 180
img_width = 180


base_dir = '/home/jamie/Documents/cacao_net/disease_dataset/'
#%% 
#Prep transfer model layers

from tensorflow.keras import layers
from tensorflow.keras import Model
# import wget
import urllib


weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_file = "inception_v3.h5"
urllib.request.urlretrieve(weights_url, weights_file)

# Instantiate the model
pre_trained_model = InceptionV3(input_shape=(img_height, img_width, 3),
                                include_top=False,
                                weights=None)

# load pre-trained weights
pre_trained_model.load_weights(weights_file)

# freeze the layers
for layer in pre_trained_model.layers:
    layer.trainable = False

# pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output




TRAINING_DIR = base_dir + 'train'
VALIDATION_DIR = base_dir + 'val'

#%% 

def fit(epochs, batch_size, rot_range):
     
     
     train_datagen = ImageDataGenerator(rescale=1./255,
                                        rotation_range=rot_range,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        fill_mode='nearest')
     
     validation_datagen = ImageDataGenerator(rescale=1./255)
     train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                        batch_size= batch_size,
                                                        class_mode='sparse',
                                                        target_size=(img_height, img_width))
     validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                                  batch_size=batch_size,
                                                                  class_mode='sparse',
                                                                  target_size=(img_height, img_width))
    
    
    
    # Flatten the output layer to 1 dimension
     x = layers.Flatten()(last_output)
    # Add a fully connected layer with 1,024 hidden units and ReLU activation
     # x = layers.Dense(1024, activation='relu')(x)
     x = layers.Dense(1024, activation='relu')(x)

    # Add a final sigmoid layer for classification
     x = layers.Dense(4, activation='softmax')(x)
    
     model = Model(pre_trained_model.input, x)
    
    
     from tensorflow.keras.optimizers import RMSprop
    # compile the model
     model.compile(optimizer=RMSprop(lr=0.0001),
                  loss='SparseCategoricalCrossentropy',
                  metrics=['acc'])
    
    # train the model (adjust the number of epochs from 1 to improve performance)
     history = model.fit(train_generator,
                validation_data=validation_generator,
                epochs=epochs,
                verbose=1)  

     fit.acc = history.history['val_acc']
     model.save('/home/jamie/Documents/cacao_net/cacao_net_model_aug')
     
     # return print('Validation acc: ' , acc)
 
#%%
#Iterate over several batch sizes and rotation ranges to find best accuracy
epochs = 1
# batch_size = [5,10,20,30,42,50,60]
# rot_range = [10, 20, 30, 40, 50, 60, 70, 80, 90]

batch_size = [10]
rot_range = [40]

batch_acc_dict = {}
rot_acc_dict = {}

fit(1, 10, 40)

# for i in batch_size:
       
#     fit(epochs, i, 40)
#     batch_acc_dict[i] = fit.acc

# for j in rot_range:
#     fit(epochs, 42, j)
#     rot_acc_dict[j] = fit.acc

# print('\nBatch size:')
# print(batch_acc_dict)
# print('\n Rotation range:')
# print(rot_acc_dict)

# best_batch = 10
# best_rot = 80

# for i in batch_size:
#     print(i , ': ', max(batch_acc_dict[i]))
    
# for i in rot_range:
#     print(i , ': ', max(rot_acc_dict[i]))


#%%
# start = datetime.now()                                
# # Train the Model
# # history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)
# #, callbacks=[callbacks])

# end = datetime.now()
# runtime= end-start
# print ("Runtime: " + str(runtime))


# model.save('/home/jamie/Documents/cacao_net/cacao_net_model_aug')
# # model.summary()
#%%
# Plot the chart for accuracy and loss on both training and validation
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
acc = history.history['acc']# Your Code Here
val_acc = history.history['val_acc']# Your Code Here
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
class_names = ['Black pod rot', 'Frosty pod rot', 'Healthy', 'Witches Broom Disease']

# img_path = '/home/jamie/Downloads/Sporulating-lesions-of-frosty-pod-rot-of-cocoa-caused-by-M-roreri-Costa-Rica-C-J_Q320.jpg'
# img_path = '/home/jamie/Downloads/c0168421-400px-wm.jpg'
# img_path = '/home/jamie/Documents/cacao_net/disease_dataset (another copy)/fpr/55805a0f34c29.jpg'
# img_path = '/home/jamie/Downloads/download(1).jpeg'
# img_path = '/home/jamie/Documents/cacao_net/disease_dataset/train/bpr/1317029-PPT.jpg'
# img_path = '/home/jamie/Documents/cacao_net/disease_dataset/train/fpr/5533023-PPT.jpg'
# img_path = '/home/jamie/Documents/cacao_net/disease_dataset/train/healthy/images358.jpg'
img_path = '/home/jamie/Documents/cacao_net/disease_dataset/train/wbd/0415-pink-mushrooms296x460.jpg'

# model = tf.keras.models.load_model('/home/jamie/Documents/cacao_net/cacao_net_model_aug')

tf.keras.metrics.Accuracy(
    name='accuracy', dtype=None
)

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