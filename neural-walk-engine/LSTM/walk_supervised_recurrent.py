import numpy as np
import pandas as pd
import time
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.layers import LeakyReLU, LSTM
from keras.optimizers import Adam
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.callbacks import LearningRateScheduler, TensorBoard

from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.python.client import device_lib
import argparse

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
  df = pd.DataFrame(data)
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
  agg = pd.concat(cols, axis=1)
  agg.columns = names
  # drop rows with NaN values
  if dropnan:
    agg.dropna(inplace=True)
  return agg


#Hyperparameters

start_time = time.time()

INPUT_SIZE = 17
OUTPUT_SIZE = 14
LEARNING_RATE = 3e-4
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-08
LOSS_FUNCTION = 'mean_absolute_error'
EPOCHS = 200
EPOCHS_DROP = 50
BATCH_SIZE = 128
DROP = 0.8
filename = 'walk_supervised_recurrent'


#Dataset processing

print("Dataset processing...")

dataset = pd.read_csv('itandroids_walk_single.txt', delimiter = "::", engine = 'python')
dataset = dataset.dropna(axis=0, how='any')
dataset = dataset.filter(items=['vx','vy','v_theta','curr_leftShoulderPitch','curr_rightShoulderPitch', 'curr_leftHipYawPitch', 
							'curr_leftHipRoll', 'curr_leftHipPitch', 'curr_leftKneePitch', 'curr_leftFootPitch', 'curr_leftFootRoll',
							'curr_rightHipYawPitch', 'curr_rightHipRoll', 'curr_rightHipPitch', 'curr_rightKneePitch', 'curr_rightFootPitch',
							'curr_rightFootRoll', 'desired_leftShoulderPitch', 'desired_rightShoulderPitch', 'desired_leftHipYawPitch', 
							'desired_leftHipRoll', 'desired_leftHipPitch', 'desired_leftKneePitch', 'desired_leftFootPitch', 'desired_leftFootRoll', 
							'desired_rightHipYawPitch','desired_rightHipRoll','desired_rightHipPitch','desired_rightKneePitch','desired_rightFootPitch',
							'desired_rightFootRoll'])

n_prev_obs = 10
train_X = dataset.values[:, :17]
train_Y = dataset.values[:, 17:]
train_X = series_to_supervised(train_X, n_in = n_prev_obs, n_out = 0, dropnan=False).fillna(0).values
train_X = train_X.reshape((train_X.shape[0], n_prev_obs, train_X.shape[1] // n_prev_obs))

#Neural Network Design
# model.add(Dense(LAYER_SIZE, input_dim = INPUT_SIZE, activation = ACTIVATION))
# for i in range(N_LAYERS - 2):
#         model.add(Dense(LAYER_SIZE, input_dim = INPUT_SIZE, activation = ACTIVATION))
# model.add(Dense(OUTPUT_SIZE))
model = Sequential()
model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(OUTPUT_SIZE))
#model.load_weights(filename)
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

tbCallBack = TensorBoard(log_dir='./folder_' + filename, histogram_freq=0, write_graph=True, write_images=True)

adam = Adam(lr=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)
model.compile (loss = LOSS_FUNCTION, optimizer = adam, metrics = ['mse', 'mae'])
history_callback = model.fit (train_X, train_Y, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = 2, callbacks=[lrate, tbCallBack], validation_split=0.1)
model.save_weights(filename)


#Save the model
print (K.get_session().graph)
model.save(filename + "_graph")
