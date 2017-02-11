import numpy as np
import matplotlib.pyplot as plt
data = [164,158,172,153,144,156,189,163,134,159,143,176,177,162,141,151,182,185,171,152]
print ("Mean: ",np.mean(data))
print ("STD: ",np.std(data))
print ("Median: ",np.median(data))
hist1, edges1 = np.histogram(data)
plt.bar(edges1[:-1], hist1, width=edges1[1:]-edges1[:-1])