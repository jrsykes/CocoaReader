import os
import json
import pandas as pd

# Specify the directory where your JSON files are stored
directory = '/jmain02/home/J2AD016/jjw02/jjs00-jjw02/dat/models/HypSweep/DisNet-Nano'

# Initialize an empty list to store the dictionaries
dict_list = []

# Loop through each file in the directory
for filename in os.listdir(directory):
    # Check if the file is a JSON file
    if filename.endswith('.json'):
        # Construct the full file path
        filepath = os.path.join(directory, filename)
        # Open the file and load the JSON data
        with open(filepath, 'r') as f:
            data = json.load(f)
            # Append the dictionary to the list
            dict_list.append(data)

# Convert the list of dictionaries into a pandas DataFrame
df = pd.DataFrame(dict_list)
#sort my column
df = df.sort_values(by=['f1'], ascending=False)

# Print the DataFrame
print(df)
