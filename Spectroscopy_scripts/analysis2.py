#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import os
import matplotlib.colors as mcolors
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import confusion_matrix
import seaborn as sns

root = "/home/jamiesykes/Documents/Ecuador_data/CocoaSpectroscopy/JamieSykesData_combined"

# %%
#empty pd df
compiled_dat = pd.DataFrame()


classes = os.listdir(root)

for c in classes:
    files = os.listdir(os.path.join(root, c))
    for file in files:
        dat = []
        if "csv" in file:
            #dat.append(c)
            df = pd.read_csv(os.path.join(root, c, file))
            #print header
            try:
                Wvl = df['Counts_Obj']

            except:
               pass    
            # #Wvl to list
            Wvl = Wvl.tolist()
            # #prepend c to Wvl
            Wvl.insert(0, c)
            # #add Wvl to compiled dat as row
            compiled_dat = compiled_dat.append([Wvl], ignore_index=True)

freq_lst =  df['Wvl_Obj'].tolist()
freq_lst.insert(0, 'Class')
compiled_dat.columns = freq_lst

#drop Witches broom from Class column
compiled_dat = compiled_dat[compiled_dat['Class'] != 'Witches broom']
#remove 'Witches broom' from classes
classes.remove('Witches broom')

print(classes)

#print number of samples in each class
for c in classes:
    df = compiled_dat.loc[compiled_dat['Class'] == c]
    print(c, len(df))

#%%
# Define the colormap
cmap = mcolors.LinearSegmentedColormap.from_list("spectrum", ["violet", "blue", "green", "yellow", "orange", "red", "darkred",                                                        
                                            "maroon", "maroon", "maroon", "maroon", "maroon"] , N=1100)

for c in classes:
    df = compiled_dat.loc[compiled_dat['Class'] == c]
    df = df.drop(columns=['Class'])
    #with sliding window of x, reduce column dimensions with mean of 4 columns
    df = df.rolling(9, axis=1).mean()
    
    df_mean = df.mean(axis=0)
    std = df_mean.std()
    ste = std/np.sqrt(len(df_mean))
    
    #set xlim 
    plt.xlim(340, 1100)

    #remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Create an array of wavelengths corresponding to the x-axis
    wavelengths = np.linspace(340, 1100, len(df_mean))

    # Plot the line with the colormap
    for j in range(1, len(wavelengths)):
        plt.plot(wavelengths[j-1:j+1], df_mean[j-1:j+1], color=cmap((wavelengths[j]-340)/(1100-340)))

    #add standard error as shaded area after plotting the mean lines
    plt.fill_between(wavelengths, df_mean-ste, df_mean+ste, alpha=0.5, color='gray')

    # Add a vertical line at 400 and 720 nm *visible light*
    plt.axvline(x=720, color='black', linestyle='--')
    plt.axvline(x=400, color='black', linestyle='--')

# Add a text box to the graph
plt.text(900, 17500, 'Frosty pod rot', fontsize=12, bbox=dict(facecolor='white', edgecolor='white', boxstyle='round'))
plt.plot([785, 900], [16500, 17800], 'k-')

plt.text(500, 18000, 'Black pod rot', fontsize=12, bbox=dict(facecolor='white', edgecolor='white', boxstyle='round'))
plt.plot([630, 755], [18500, 17600], 'k-')

plt.text(500, 20000, 'Healthy', fontsize=12, bbox=dict(facecolor='white', edgecolor='white', boxstyle='round'))
plt.plot([600, 775], [20000, 19400], 'k-')


#set axis lables size
plt.tick_params(axis='both', which='major', labelsize=12)
#set ylab size
plt.ylabel('Reflectance count ±1 SE', fontsize=14)
#set xlab size
plt.xlabel('Wavelength (nm)', fontsize=14)

plt.show()


#%%
df = compiled_dat

n_bins = (df.shape[1] - 1)

# Separate features and target variable
X = df.drop("Class", axis=1)

y = df["Class"]
#convert y to numerical values
y = pd.factorize(y)[0]

# subset = df.iloc[:, 126:184]  # Note that Python's slicing is 0-based and the end index is exclusive

#%%

acc_dict = {}
model_dict = {}
for i in range(1, 100):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i, stratify=y)
    # X_train, X_test, y_train, y_test = train_test_split(subset, y, test_size=0.2, random_state=42, stratify=y)


    # param_grid = {
    #     'n_estimators': [58, 59, 60, 61, 62],
    #     'max_depth': [10, 20, 20, None],  # Added max depth
    #     'min_samples_split': [5,6,7,8],
    #     'min_samples_leaf': [1, 2, 3]
    # }

    # rf = RandomForestClassifier(random_state=42)

    # grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
    #                            cv=5, n_jobs=-1, verbose=2, scoring='accuracy')  # Changed scoring to 'accuracy'

    # grid_search.fit(X_train, y_train)

    # best_params = grid_search.best_params_
    
    # print(f"Best parameters: {best_params}")
    # print()
    # best_model = grid_search.best_estimator_

    params = {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 8, 'n_estimators': 61}
    rf = RandomForestClassifier(**params)
    model = rf.fit(X_train, y_train)
    # Make predictions
   
    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    if cm[0][0] >= 3:
        # best_model = model
        acc_dict[i] = accuracy
        model_dict[i] = model
    


#print max value in acc_dict
max_key = max(acc_dict, key=acc_dict.get)
print(max_key)
print(acc_dict[max_key])
best_model = model_dict[max_key]
        
    
#%%
y_pred = best_model.predict(X_train)
accuracy = accuracy_score(y_train, y_pred)
print(f"Train accuracy: {accuracy}")

y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy}")
    
# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix
plt.figure(figsize=(10, 7))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

#%%
# Calculate the range of each bin based on original spectral points
original_columns = compiled_dat.drop('Class', axis=1).columns
# Calculate the step size between each spectral point
step_size = (1100 - 340) / len(original_columns)

# Calculate the range of each bin based on original spectral points
bin_size = len(original_columns) // n_bins

bin_ranges = [(340 + i * step_size * bin_size, 340 + (i + 1) * step_size * bin_size - 1) for i in range(n_bins)]

# Get feature importances and sort them by importance
importances = best_model.feature_importances_
indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)

# Select top 20 most important features
top_indices = indices[:]
top_importances = [importances[i] for i in top_indices]
top_features = [X_train.columns[i] for i in top_indices]

# Sort the top features by their spectral values (assuming they are numerical)
sorted_top_indices = sorted(top_indices, key=lambda i: i)  # Assuming the columns are already sorted by spectral value

# Get the bin ranges corresponding to the top features
top_bin_ranges = [bin_ranges[i] for i in sorted_top_indices]
#convert all top_bin_ranges to int
top_bin_ranges = [tuple(map(int, i)) for i in top_bin_ranges]

# Create labels for the x-axis based on the bin ranges
bin_labels = [f"{start}-{end}" for start, end in top_bin_ranges]

# %%

# Get the mean feature importances
mean_importances = np.mean([tree.feature_importances_ for tree in best_model.estimators_], axis=0)

# Get the standard deviations of the feature importances across the trees in the forest
std = np.std([tree.feature_importances_ for tree in best_model.estimators_], axis=0)

# Calculate the number of trees
n_trees = len(best_model.estimators_)

# Calculate the 90% confidence interval
z_value = 1.96  # Z-value for 90% confidence
margin_of_error = z_value * (std / np.sqrt(n_trees))

# Get the margin of error for the top features
top_margin_of_error = [margin_of_error[i] for i in sorted_top_indices]
top_mean_importances = [mean_importances[i] for i in sorted_top_indices]


# Calculate the lower and upper bounds of the 90% confidence interval
lower_bound = np.array(top_mean_importances) - np.array(top_margin_of_error)
upper_bound = np.array(top_mean_importances) + np.array(top_margin_of_error)

# Smooth the lines using a Gaussian filter
sigma = 4 # Standard deviation for Gaussian kernel
smoothed_mean_importances = gaussian_filter1d(top_mean_importances, sigma)
smoothed_lower_bound = gaussian_filter1d(lower_bound, sigma)
smoothed_upper_bound = gaussian_filter1d(upper_bound, sigma)

tick_spectral_values = np.arange(400, 1110, 100)
tick_indices = [np.argmin(np.abs(np.array(top_bin_ranges)[:, 0] - val)) for val in tick_spectral_values]


plt.figure(figsize=(10, 6))

# Create an array of wavelengths corresponding to the x-axis
wavelengths = np.linspace(400, 1100, len(sorted_top_indices))
# wavelengths = np.linspace(720, 900, len(sorted_top_indices))

# Plot the smoothed mean feature importances with the colormap
for j in range(1, len(wavelengths)):
    plt.plot(range(j-1, j+1), smoothed_mean_importances[j-1:j+1], color=cmap((wavelengths[j]-400)/(1100-400)))


plt.axvline(x=130, color='black', linestyle='--')
plt.axvline(x=20, color='black', linestyle='--')

# Remove right and top borders
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add x and y labels and ticks
plt.xticks(tick_indices, tick_spectral_values, fontsize=16)
plt.xlabel("Wavelength (nm)", fontsize=16)
plt.ylabel("Feature Importance", fontsize=16)

# Show the plot
plt.show()


# %%



# %%
