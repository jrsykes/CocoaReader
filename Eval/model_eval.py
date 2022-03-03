import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix




# with open('/home/jamiesykes/models/ResDesNet50V2_model.yaml', 'r') as f:
#     model = tf.keras.models.model_from_yaml(f)

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

model = tf.keras.models.load_model('/home/jamiesykes/models/ResDesNet50V2_model_bias_imagenet__subset53_10-2-22.h5', custom_objects=dependencies)

val_dir = '/home/jamiesykes/Documents/ResDesNet50/TestData/val'
val_dir_subset = '/home/jamiesykes/Documents/ResDesNet50/TestData/val_subset53'
#%% Load data
img_height = 300
img_width = 300

batch_size = 42

testData = image_dataset_from_directory(
    directory = val_dir,
    labels = 'inferred',
    label_mode = 'categorical',
    image_size=(img_height, img_width),
    batch_size=batch_size)

#%% Sanity check, classify an image


from PIL import Image

from keras.applications.resnet50 import preprocess_input


size = 300,300
img = Image.open('/home/jamiesykes/Documents/ResDesNet50/TestData/val/GrapesDiseased/GrapesDiseased3.jpg')
img.thumbnail(size)

plt.imshow(img)


img = img.resize((300,300), Image.ANTIALIAS)

img = np.asarray(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

preds = model.predict(img)
print(sorted(os.listdir(val_dir))[np.argmax(preds)])
print(np.max(preds))

#%%
# label_names =  np.array([])
# for x, y in testData:
#   label_names = np.concatenate([label_names, np.argmax(y.numpy(), axis=-1)])


# def predict_class_label_number(dataset):
#   """Runs inference and returns predictions as class label numbers."""
#   rev_label_names = {l: i for i, l in enumerate(label_names)}
#   return [
#       rev_label_names[o[0][0]]
#       for o in model.predict_top_k(dataset, batch_size=128)
#   ]

# def show_confusion_matrix(cm, labels):
#   plt.figure(figsize=(10, 8))
#   sns.heatmap(cm, xticklabels=labels, yticklabels=labels, 
#               annot=True, fmt='g')
#   plt.xlabel('Prediction')
#   plt.ylabel('Label')
#   plt.show()



#%% Create and save confusion matrix

predictions = np.array([])
labels =  np.array([])
images = []
for x, y in testData:
  predictions = np.concatenate([predictions, np.argmax(model.predict(x), axis=-1)])
  labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])

cm = confusion_matrix(labels, predictions)

names = sorted(os.listdir(val_dir))

df = pd.DataFrame(cm,columns=names)
df.insert(loc=0,column='/',value=names)


df.to_csv('confusion_with_aug+bias+imagenet_weights_subset53.csv')
#%% Calculate per class accuracey

df = pd.DataFrame(cm,columns=names)
tp = pd.DataFrame(df.values[[np.arange(df.shape[0])]*2], columns=['tp'], dtype='float')

tp = (tp['tp'].to_list())


totals = df.sum(axis=1).to_list()


labels = sorted(os.listdir(val_dir))
results = pd.DataFrame(totals, columns=['totals'], index=labels)
results['tp'] = tp
results['acc'] = round(results['tp']/results['totals'],2)
# results = results.sort_values(by=['acc'])

print(results)


#%% Get loss and accuracey

print(model.evaluate(testData))

# print("Model accuracy: ", str(100 * acc))
# print('Loss: ' + str(loss))
# print('Accuracey: ' + str(acc))
# print('F1: ' + str(f1_metric))