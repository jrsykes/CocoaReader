#%%
import json
import wandb
import os

mean_train_metrics_dict = {'loss': 0.07613806964363903, 'f1': 0.9977126328955597, 'acc': 0.9956349206349205, 'precision': 1.0, 'recall': 0.9956349206349205}                                                                                                                              
mean_val_metrics_dict = {'loss': 1.5738358855247498, 'f1': 0.10796883671883672, 'acc': 0.08928571428571427, 'precision': 0.2425595238095238, 'recall': 0.08928571428571427}    

project_name = 'IR-RGB_sweep'
run_name = 'RGB_cross-val'

run = wandb.init(project=project_name)
artifact = wandb.Artifact(project_name + '_results', type='dataset')
# Log the results as wandb artifacts
results_dict = {'train_metrics': mean_train_metrics_dict, 'val_metrics': mean_val_metrics_dict}

with open(run_name + 'results_dict.json', 'w') as f:
    json.dump(results_dict, f)

#artifact.add_file(run_name + 'results_dict.json')
#run.log_artifact(artifact)
wandb.finish()


os.remove(run_name + 'results_dict.json')
#%%