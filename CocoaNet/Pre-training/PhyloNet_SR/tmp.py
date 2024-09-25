import RobinsonFoulds
import torch
import pandas as pd
import os



labels = torch.tensor([ 70,  60,  14,  38, 100,  27,  17,  64,   6,   3,  34,  60,  75,   3,
         21,  96,  48,  11,  90,  72,  38,  77,  21,  88,  22,  96,  40,  50,
         17,  75,  24,  51,  29,  33,  41,  23,  14,  71,  75,  87,  94,  73,
         29,  86,  87,  72,  86,  50,  23,   2,  22,   7,  45,  79,  56,  24,
         89,  33,  39,  27,  77,  19,  30,  33,  39,  46,  86,  27,  42,  44,
         94,  12,  59,  51,  26,  30,  16,  52,  38, 100,  36,  74,  32,  30,
        100,  96,  60,  78,  38,  95,  46,  70,  35,  81,  47,  42,  77,  30,
          2,  33])


taxonomy = pd.read_csv('/users/jrs596/scratch/dat/flowers102_split/flowers_taxonomy.csv', header=0)

#tensor with random values, size [100,152]
encoded_pooled = torch.rand(100, 152)


trees = RobinsonFoulds.trees(taxonomy, labels, encoded_pooled)

print(trees["input_tree"])
print()
print(trees["output_tree"])