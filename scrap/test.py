import os
import shutil

base_dir = '/local/scratch/jrs596/dat/Forestry_ArableImages_GoogleBing_Licenced_clean'


for i in os.listdir(base_dir):
  sub_dir = os.path.join(base_dir, i)
  count = 0
  for j in os.listdir(sub_dir):
    source = os.path.join(sub_dir, j)
    #os.makedirs(os.path.join(base_dir + '_unorganised', i), exist_ok = True)
    dest = os.path.join(base_dir + '_unorganised/test', i + str(count) + '.jpg')
    os.link(source, dest)
    count += 1