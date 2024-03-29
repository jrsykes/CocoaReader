import requests
import os 
import pandas as pd
import time

subscription_key = ""
search_url = "https://api.bing.microsoft.com/v7.0/images/search"
headers = {"Ocp-Apim-Subscription-Key" : subscription_key}
base_dir = '/local/scratch/jrs596/dat/GoogleBingCocoaImages/'
df = pd.read_csv('/local/scratch/jrs596/dat/cocoa_disease_search_terms.csv', header=None)


def search(search_key):
    params  = {"q": search_key, "imageType": "photo", "count": "1000"}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    return search_results['value']



def saver(image_path, images):
    for i in images:
        url = i['contentUrl']
        image_id = i['imageId']
        image = requests.get(url, allow_redirects=True)
        filename = image_id + str(time.time()) + '.jpeg'

        if os.path.exists(image_path) == False:
                os.makedirs(image_path)
        
        with open(os.path.join(image_path, filename), 'wb') as f:
            f.write(image.content)

         

#%%

species_set = set(df.iloc[:,0])


for j in species_set:
    subset = pd.DataFrame(columns = list(range(3)))
    for row in df.iterrows():
        if j in row[1].values.tolist():
            
            subset.loc[len(subset)] = row[1].values.tolist()

    for state in ['Healthy']:
        if state == 'Healthy':
            #search_keys = [subset.iloc[0,1], subset.iloc[0,2]]
            search_keys = subset.iloc[:,1:].values.tolist()
            for key in search_keys:
                image_path = os.path.join(base_dir, j)   
                images = search(key[0] + key[1])
                saver(image_path, images)

        if state == 'Diseased' and subset.iloc[:,2].to_string() != '0    .' and subset.iloc[:,3].to_string() != '0    .':
            key_list = list(set([item for sublist in subset.values.tolist() for item in sublist]))
            
            for k in range(2):
                key1 = key_list[k]
                for combination in ['disease', 'fungal', 'pathogen']:
                    seach_key = key1 + ' ' + combination
                 
                    image_path = os.path.join(base_dir, (j + state).replace(' ', '_')) 
                    #images = search(seach_key)
                    #saver(image_path, images)



                    
                    
                    
                    
                    
                    
                    
                    
                    
