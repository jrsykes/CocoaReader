from difPy import dif
import os

dir_ = '/local/scratch/jrs596/dat/Forestry_ArableImages_GoogleBing_clean'
for i in os.listdir(dir_):
	search = dif(os.path.join(dir_, i), delete=True, silent_del=True)
