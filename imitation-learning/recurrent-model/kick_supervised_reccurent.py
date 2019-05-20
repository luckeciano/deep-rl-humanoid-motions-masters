import numpy as np
import pandas as pd
from matplotlib import pyplot

from keras.layers import Input, Dense, LSTM
from keras.models import Model, Sequential
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.callbacks import LearningRateScheduler

from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.python.client import device_lib

#Hyperparameters

N_LAYERS = 3
LAYER_SIZE = 64
ACTIVATION = 'tanh'
INPUT_SIZE = 1
OUTPUT_SIZE = 23
LEARNING_RATE = 0.01
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-08
LOSS_FUNCTION = 'mean_squared_error'
EPOCHS = 10000
EPOCHS_DROP = 100
BATCH_SIZE = 77
DROP = 0.75

# series_to_supervised helper


from pandas import DataFrame
from pandas import concat
 
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

#Dataset processing

n_prev_obs = 20

dataset = pd.read_csv('kick_itandroids.txt', delimiter = "::", engine = 'python')
dataset = dataset.dropna(axis=0, how='any')
values = dataset.values[0:, 1:]

train = series_to_supervised(values[0:72, :], n_in = n_prev_obs, dropnan = False).fillna(0).values
for i in range(10):
	print(i)
	train_i = series_to_supervised(values[i*72:(i+1)*72, :], n_in = n_prev_obs, dropnan = False).fillna(0).values
	train = np.vstack((train,train_i))

train_X, train_Y = train[:, :n_prev_obs*23], train[:, -23:]

train_X = train_X.reshape((train_X.shape[0], n_prev_obs, train_X.shape[1] // n_prev_obs))

print(train_X.shape)


#Neural Network Design

# design network
model = Sequential()
model.add(LSTM(25, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(23))


adam = Adam(lr=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)
model.compile (loss = LOSS_FUNCTION, optimizer = adam, metrics = ['mse', 'mae'])
model.summary()

#Training Procedure

def step_decay(epoch):
   initial_lrate = LEARNING_RATE
   drop = DROP
   epochs_drop = float(EPOCHS_DROP)
   lrate = initial_lrate * np.power(drop,  
           np.floor((1+epoch)/epochs_drop))
   return lrate

lrate = LearningRateScheduler(step_decay)

history_callback = model.fit (train_X, train_Y, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = 1, callbacks=[lrate], shuffle = False)



#Save the model
print (K.get_session().graph)
model.save('neural_kick_lstm')

# plot history
pyplot.plot(history_callback.history['loss'], label='train')
pyplot.legend()
pyplot.show()

