import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import pandas as pd



def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


dependencies = {
    'f1_metric': f1_metric
}

model = tf.keras.models.load_model('/local/scratch/jrs596/ResNetFung50/models/ResDesNet50V2_model_bias_4-2-22.h5', custom_objects=dependencies)

img_height = 1000
img_width = 1000

batch_size = 42

testData = image_dataset_from_directory(
    directory = '/local/scratch/jrs596/dat/PlantNotPlant_TinyIM_filtered_split/val',
    labels = 'inferred',
    label_mode = 'categorical',
    image_size=(img_height, img_width),
    batch_size=batch_size)


predictions = np.array([])
labels =  np.array([])
images = []
for x, y in testData:
  predictions = np.concatenate([predictions, np.argmax(model.predict(x), axis=-1)])
  labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(labels, predictions)

names = sorted(os.listdir('/local/scratch/jrs596/dat/PlantNotPlant_TinyIM_filtered_split/val'))

df = pd.DataFrame(cm,columns=names)
df.insert(loc=0,column='/',value=names)
print(df)

df.to_csv('/local/scratch/jrs596/ResNetFung50/confusion/ResDes18_1kdim_HighRes_TinyIN_Filtered.csv')