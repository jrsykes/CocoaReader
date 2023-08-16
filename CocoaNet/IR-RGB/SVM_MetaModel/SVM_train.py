#%%
import pandas as pd
from sklearn.svm import SVC
from joblib import dump
import numpy as np
import torch
#%%
# Load prediction data from CSV files
efficientnet_b0_data = pd.read_csv('/users/jrs596/scratch/dat/cross_val_predictions/efficientnet_b0.csv')
DisNet_pico_data = pd.read_csv('/users/jrs596/scratch/dat/cross_val_predictions/DisNet_picoIR.csv')

# Verify that the true labels in both files are identical
assert all(efficientnet_b0_data.iloc[:, 0] == DisNet_pico_data.iloc[:, 0]), "Mismatch in true labels between the two files"

#%%

# Extract true labels from the first column of either CSV file
true_labels = efficientnet_b0_data.iloc[:, 0].values

# Extract model output values from columns 2-5 of both CSV files
efficientnet_b0_preds = efficientnet_b0_data.iloc[:, 1:5].values
DisNet_pico_preds = DisNet_pico_data.iloc[:, 1:5].values

# Find the index of the maximum value in each row for both arrays
efficientnet_b0_max_indices = np.argmax(efficientnet_b0_preds, axis=1)
DisNet_pico_max_indices = np.argmax(DisNet_pico_preds, axis=1)

print(efficientnet_b0_max_indices.shape)
print(DisNet_pico_max_indices.shape)
#%%
# Stack predictions to form feature vectors for the SVM
X_train = np.column_stack((efficientnet_b0_max_indices, DisNet_pico_max_indices))

print(X_train.shape)
#%%
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)



#%%
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid']}
grid = GridSearchCV(SVC(probability=True), param_grid, refit=True, verbose=3)
grid.fit(X_train, true_labels)
print(grid.best_params_)
svm = grid.best_estimator_

# Train the SVM
# svm = SVC(probability=True, C=10, gamma=0.01, kernel='sigmoid')
# svm.fit(X_train_scaled, true_labels)

# Save the trained SVM model
dump(svm, '/users/jrs596/scratch/dat/cross_val_predictions/svm_meta_model.joblib')

print("SVM Meta-Model Trained and Saved")
# %%
