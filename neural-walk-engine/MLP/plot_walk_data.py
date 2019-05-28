from mpl_toolkits.mplot3d import Axes3D 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv
dataset = pd.read_csv('itandroids_walk.txt', delimiter = "::", engine = 'python')
fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)
for x in dataset:
	dataset[x].hist(bins=100)
	plt.title(x)
	plt.show()
#dataset.hist()
# dataset['vx'].hist()
# dataset['vy'].hist()
# ax.scatter(dataset['vx'].values,dataset['vy'].values, alpha = 0.1)
# ax.set_xlabel('Vx')
# ax.set_ylabel('Vy')
# #ax.set_zlabel('Vtheta')
# ax.legend(loc='best')


# dataset_1 = dataset.loc[dataset['vx'] > 0.1]
# dataset_2 = dataset.loc[dataset['vx'] < -0.1]
# plt.figure(2)
# dataset_3 = dataset.loc[(dataset['vx'] < 0.01) & (dataset['vx'] > -0.01)]
# dataset_3 = dataset_3.sample(1000)

# dataset_4 = dataset.loc[(dataset['vx'] < 0.1) & (dataset['vx'] > 0.01)]
# dataset_5 = dataset.loc[(dataset['vx'] > -0.1) & (dataset['vx'] < -0.01)]
# dataset_45 = pd.concat([dataset_4, dataset_5])
# dataset_45 = dataset_45.sample(10000)
# dataset_final = pd.concat([dataset_1, dataset_2, dataset_3, dataset_45])
# dataset_final['vx'].hist(bins=100)
#dataset_3['vx'].hist(bins=100)

plt.show()