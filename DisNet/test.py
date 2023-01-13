#%%
#read yaml file as dictionary
import yaml
with open('CocoaNetSweepConfig.yml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

print(config['parameters'])

#%%