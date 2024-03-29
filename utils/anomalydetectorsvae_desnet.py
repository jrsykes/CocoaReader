# -*- coding: utf-8 -*-
"""AnomalyDetectorsVAE-DesNet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WgoM2n_GPZBAvHDOpZc4vrVHRS1aynSW
"""

# Commented out IPython magic to ensure Python compatibility.
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.losses import mse, binary_crossentropy, kl_divergence
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from ImageDataAugmentor.image_data_augmentor import *


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer

#import seaborn as sns
import matplotlib.pyplot as plt
from albumentations import (
    Compose, RandomBrightness, JpegCompression, HueSaturationValue, RandomContrast, HorizontalFlip,
    Rotate
)

import os
#######################
img_height = 32
img_width = 32
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

data_generator_no_aug = ImageDataGenerator()

X_train = data_generator_no_aug.flow_from_directory(
    '/local/scratch/jrs596/dat/ResNetFung50+_images_unorganised',
    target_size=(img_height, img_width),
    batch_size=batch_size)
    #class_mode='categorical')#,
    #class_names = classes)


X_test = data_generator_no_aug.flow_from_directory('/local/scratch/jrs596/dat/Forestry_ArableImages_GoogleBing_Licenced_clean_unorganised',
    target_size=(img_height, img_width),
    batch_size=batch_size)
    #class_mode='categorical',
    #class_names = classes)


#######################


# %matplotlib inline

#urls = [
#        "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz",
#        "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names"
#        ]#

## this pre-processing code of the KDD dataset is adapter from https://github.com/lironber/GOAD/blob/master/data_loader.py#

#df_colnames = pd.read_csv(urls[1], skiprows=1, sep=':', names=['f_names', 'f_types'])
#df_colnames.loc[df_colnames.shape[0]] = ['status', ' symbolic.']#

#df = pd.read_csv(urls[0], header=None, names=df_colnames['f_names'].values)
#df_symbolic = df_colnames[df_colnames['f_types'].str.contains('symbolic.')]
#df_continuous = df_colnames[df_colnames['f_types'].str.contains('continuous.')]
#samples = pd.get_dummies(df.iloc[:, :-1], columns=df_symbolic['f_names'][:-1])#

#labels = np.where(df['status'] == 'normal.', 1, 0)#

#scaler = MinMaxScaler()
#df_scaled = scaler.fit_transform(samples)#

#norm_samples = df_scaled[labels == 1]  # normal data
#attack_samples = df_scaled[labels == 0]  # attack data#

#norm_labels = labels[labels == 1]
#attack_labels = labels[labels == 0]#

#attack_samples.shape#

## generate train set
## training set will consist of the normal ds#

#len_norm = len(norm_samples)
#len_norm_train = int(0.8 * len_norm)
#X_train = norm_samples[:len_norm_train]#

## generate test set consist of 50% attack and 50% normal#

#X_test_norm = norm_samples[len_norm_train:]
#len_attack_test = len(X_test_norm) # we will use the same number
#X_test_attack = attack_samples[:len_attack_test]#

#X_test = np.concatenate([X_test_norm, X_test_attack])
#y_test = np.ones(len(X_test))
#y_test[:len(X_test_norm)] = 0


def get_error_term(v1, v2, _rmse=True):
    if _rmse:
        return np.sqrt(np.mean((v1 - v2) ** 2, axis=1))
    #return MAE
    return np.mean(abs(v1 - v2), axis=1)

# The reparameterization trick

def sample(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

#original_dim = X_train.shape[1]
#input_shape = (original_dim,)

original_dim = (None, img_height, img_width, 3, batch_size)
input_shape = (None, img_height, img_width, 3, batch_size)
flattened_shape = (None, img_height * img_width * 3 * batch_size)

intermediate_dim = int((img_height/2 * img_width/2 * 3 * batch_size))
latent_dim = int((img_height/3 * img_width/3 * 3 * batch_size))

print('\nOriginal dim: ' + str(original_dim))
print('input_shape : ' + str(input_shape))
print('intermediate_dim: ' + str(intermediate_dim))
print('latent_dim : ' + str(latent_dim) + '\n')


encoder = tf.keras.Sequential()
layer_1 = Dense(shape=input_shape, name='encoder_input')
encoder.add(layer_1)

layer_2 = tf.keras.layers.Flatten()
encoder.add(layer_2)

layer_2.input_shape (original_dim) 

layer_2.output_shape (flattened_shape) 


print(encoder.output_shape)

exit(0)
## encoder model
inputs = Input(shape=input_shape, name='encoder_input')

#x = tf.keras.layers.Flatten(inputs)
#print(x)
#exit(0)


x = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)
# use the reparameterization trick and get the output from the sample() function
z = Lambda(sample, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
encoder = Model(inputs, z, name='encoder')
encoder.summary()

exit(0)

# decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)
# Instantiate the decoder model:
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# full VAE model
outputs = decoder(encoder(inputs))
vae_model = Model(inputs, outputs, name='vae_mlp')


# the KL loss function:
def vae_loss(x, x_decoded_mean):
    # compute the average MSE error, then scale it up, ie. simply sum on all axes
    reconstruction_loss = K.sum(K.square(x - x_decoded_mean))
    # compute the KL loss
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.square(K.exp(z_log_var)), axis=-1)
    # return the average loss over all 
    total_loss = K.mean(reconstruction_loss + kl_loss)    
    #total_loss = reconstruction_loss + kl_loss
    return total_loss

opt = optimizers.Adam(learning_rate=0.0001, clipvalue=0.5)
#opt = optimizers.RMSprop(learning_rate=0.0001)

vae_model.compile(optimizer=opt, loss=vae_loss)
vae_model.summary()


# Finally, we train the model:
results = vae_model.fit(X_train, X_train,
                        shuffle=True,
                        epochs=32)#,
                        #batch_size=batch_size)

exit(0)

plt.plot(results.history['loss'])
#plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right');
plt.show()

X_train_pred = vae_model.predict(X_train)

mae_vector = get_error_term(X_train_pred, X_train, _rmse=False)
print(f'Avg error {np.mean(mae_vector)}\nmedian error {np.median(mae_vector)}\n99Q: {np.quantile(mae_vector, 0.99)}')
print(f'setting threshold on { np.quantile(mae_vector, 0.99)} ')

error_thresh = np.quantile(mae_vector, 0.99)

X_pred = vae_model.predict(X_test)
mae_vector = get_error_term(X_pred, X_test, _rmse=False)
anomalies = (mae_vector > error_thresh)

np.count_nonzero(anomalies) / len(anomalies)

from sklearn.metrics import classification_report

print(classification_report(y_test, anomalies))

X_pred.shape

X_encoded = encoder.predict(X_test)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_transform = pca.fit_transform(X_encoded)

#plt.figure(figsize=(12, 10))
#sns.scatterplot(x=X_transform[:, 0], y=X_transform[:, 1], s=20, hue=mae_vector)
#plt.grid()
#plt.show()#

#plt.figure(figsize=(12, 10))
#sns.scatterplot(x=X_transform[:, 0], y=X_transform[:, 1], s=20, hue=anomalies)
#plt.grid()
#plt.show()#

#plt.figure(figsize=(12, 10))
#sns.scatterplot(x=X_transform[:, 0], y=X_transform[:, 1], s=10, hue=y_test)
#plt.grid()
#plt.show()

