import os
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from tensorflow.keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf

import datetime
from tensorflow.keras.optimizers import Adam

from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow.keras.backend as K
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from ImageDataAugmentor.image_data_augmentor import *
import albumentations
from albumentations import (
    Compose, RandomBrightness, JpegCompression, HueSaturationValue, RandomContrast, HorizontalFlip,
    Rotate
)
import numpy as np

### Calculate bias for final layer
dir = '/local/scratch/jrs596/dat/ResNetFung50+_images_organised/train/'
list_cats = []
for i in sorted(os.listdir(dir)):
    path, dirs, files = next(os.walk(dir + i))
    list_cats.append(len(files))
    
weights = []
for i in list_cats:
    weights.append(np.log((max(list_cats)/i)))

initial_bias = np.array(weights)
###################################

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():

    def f1_metric(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return f1_val


    img_height = 300
    img_width = 300
    batch_size = 42


    transforms = Compose([
            Rotate(limit=40),
            RandomBrightness(limit=0.1),
            JpegCompression(quality_lower=85, quality_upper=100, p=0.5),
            HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            RandomContrast(limit=0.2, p=0.5),
            HorizontalFlip(),
        ])
    train_datagen = ImageDataAugmentor(
        augment=transforms,
        preprocess_input=None)

    train_ds = train_datagen.flow_from_directory(
        '/local/scratch/jrs596/dat/ResNetFung50+_images_organised/train',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    data_generator_no_aug = ImageDataGenerator()
    val_ds = data_generator_no_aug.flow_from_directory('/local/scratch/jrs596/dat/ResNetFung50+_images_organised/val',
        target_size=(img_height, img_width),
        class_mode='categorical')


    base_model = keras.applications.resnet_v2.ResNet50V2(weights=None include_top=False, input_shape= (img_height,img_width,3))# # or weights = 'imagenet'

    x = base_model.output
    x = tf.keras.applications.resnet_v2.preprocess_input(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    predictions = Dense(66, activation= 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    adam = Adam(learning_rate=0.0001)


    model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['acc', f1_metric])
    model.layers[-1].bias.assign(initial_bias)


    log_dir = "/local/scratch/jrs596/ResNetFung50/models/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    es_callback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights='True')

    model.fit(train_ds, validation_data=val_ds, epochs = 1000, batch_size = batch_size, callbacks=[tensorboard_callback, es_callback])

    model.save('/local/scratch/jrs596/ResNetFung50/models/ResDesNet50V2_model_bias_4-2-22.h5')#

    yaml_model= model.to_yaml()
    with open('/local/scratch/jrs596/ResNetFung50/models/ResDesNet50V2_model_bias_4-2-22.yaml', 'w') as yaml_file:
        yaml_file.write(yaml_model)

