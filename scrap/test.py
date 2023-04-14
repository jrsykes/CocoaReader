#%%
# importing the module
import json
import time

#start time
start_time = time.time()
# Opening JSON file
with open('/jmain02/home/J2AD016/jjw02/jjs00-jjw02/dat/models/HypSweep/ConvNextOpt_results_1.json') as json_file:
    data = json.load(json_file)

# # Save the updated dictionary back to the JSON file
# with open('/jmain02/home/J2AD016/jjw02/jjs00-jjw02/dat/models/HypSweep/ConvNextOpt_results_test.json', 'w') as f:
#         json.dump(data, f)
#end time
end_time = time.time()

print("Time taken to run this cell :", end_time-start_time)

#%%
# print()
print(data.keys())

# [print(data['saul_run_13'])]
# #%%