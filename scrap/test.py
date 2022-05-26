from scipy.special import gamma, factorial
import numpy as np
import matplotlib.pyplot as plt
import math 

gamma = np.random.gamma(7.5,1,1000)


df = 224*224-1

chi = np.random.chisquare(df, size=1000)

gau = np.random.normal(0.5, 1, size=1000)


t = np.divide(gau,np.sqrt(chi/df))

plt.hist(gau, 50)
plt.show()