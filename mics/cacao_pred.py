#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 18:34:32 2020

@author: jamie
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

model = tf.keras.models.load_model('/home/jamie/Documents/weed_net/weed_net_model')
img_path = '/home/jamie/Documents/weed_net/weed_net_data/test/aethusa_cynapium/10110.jpg'
#%%

class_names = ['aethusa_cynapium', 'amaranthus_retroflexus', 'anagallis_arvensis', 'anchusa_arvensis', 'anthemis_arvensis', 'aphanes_arvensis', 'atriplex_patula', 'avena_fatua', 'brassica_napus_ssp._oleifera', 'capsella_bursa-pastoris', 'cardamine_hirsuta', 'cerastium_fontanum', 'chenopodium_album', 'cirsium_arvense', 'convolvulus_arvensis', 'conyza_canadensis', 'elytrigia_repens', 'epilobium_spp.', 'equisetum_arvense', 'euphorbia_peplus', 'fallopia_convolvulus', 'fumaria_officinalis', 'funaria_hygrometrica', 'galeopsis_tetrahit', 'galium_aparine', 'geranium_molle', 'glebionis_segetum', 'lamium_amplexicaule', 'lamium_purpureum', 'lapsana_communis', 'malva_sylvestris', 'marchantia_polymorpha', 'matricaria_discoides', 'mentha_arvensis', 'mercurialis _annua', 'myosotis_arvensis', 'oxalis_corniculata', 'papaver_rhoeas', 'persicaria_lapathifolia', 'persicaria_maculosa', 'poa_annua', 'polygonum', 'ranunculus_repens', 'raphanus_raphanistrum', 'reseda_lutea', 'rorippa_sylvestris', 'rumex_acetosa', 'rumex_spp', 'sagina_procumbens', 'salix_caprea', 'senecio_vulgaris', 'silene_latifolia', 'sinapis_arvensis', 'solanum_nigrum', 'sonchus_arvensis', 'sonchus_spp.', 'spergula_arvensis', 'stellaria_media', 'taraxacum', 'thlaspi_arvense', 'tripleurospermum_inodorum', 'tussilago_farfara', 'urtica_urens', 'veronica_hederifolia', 'veronica_persica', 'vicia_sativa', 'viola_arvensis']


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


