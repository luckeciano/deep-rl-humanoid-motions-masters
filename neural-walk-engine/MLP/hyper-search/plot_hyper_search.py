from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt

mypath = 'hyper-search-plot'

files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

plt.ylim(top=0.05)
sort_res = []
for file in files:
	file = join(mypath, file)
	mae = np.loadtxt(file)
	last_value = mae[-1]
	el = (last_value, file)
	sort_res.append(el)
	
sort_res = sorted(sort_res)
for i in range(30):
	file = sort_res[i][1]
	mae = np.loadtxt(file)
	t = mae.shape[0]
	plt.plot(range(t), mae, label=file)
	plt.legend(loc='best')
	plt.grid()

plt.yticks(np.arange(0, 0.05, step=0.01))
plt.show()

