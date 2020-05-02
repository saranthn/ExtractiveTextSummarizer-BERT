import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset = pd.read_csv('data.csv')
x = dataset.iloc[:, 0].values
hist, bins = np.histogram(x, bins = 4)
print(bins)
plt.hist(x, bins = 4)
plt.show()