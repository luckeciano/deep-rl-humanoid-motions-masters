import numpy as np
import pandas as pd

from keras.layers import Input, Dense
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
EPOCHS = 100000
EPOCHS_DROP = 5000
BATCH_SIZE = 128
DROP = 0.8

#Dataset processing

dataset = pd.read_csv('kick_itandroids_single.txt', delimiter = "::", engine = 'python')
dataset = dataset.dropna(axis=0, how='any')
train_X = dataset.values[0:127, 0]
train_Y = dataset.values[0:127, 1:24]


#Neural Network Design
model =  Sequential()
model.add(Dense(LAYER_SIZE, input_dim = INPUT_SIZE, activation = ACTIVATION))
for i in range(N_LAYERS - 2):
        model.add(Dense(LAYER_SIZE, input_dim = INPUT_SIZE, activation = ACTIVATION))
model.add(Dense(OUTPUT_SIZE))


model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=False)


#Training Procedure
def step_decay(epoch):
   initial_lrate = LEARNING_RATE
   drop = DROP
   epochs_drop = float(EPOCHS_DROP)
   lrate = initial_lrate * np.power(drop,  
           np.floor((1+epoch)/epochs_drop))
   return lrate

lrate = LearningRateScheduler(step_decay)

adam = Adam(lr=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)
model.compile (loss = LOSS_FUNCTION, optimizer = adam, metrics = ['mse', 'mae'])
history_callback = model.fit (train_X, train_Y, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = 2, callbacks=[lrate])
loss_hist_mse = np.array(history_callback.history["mean_squared_error"])
loss_hist_mae = np.array(history_callback.history["mean_absolute_error"])

#Saving Metrics History for plot
np.savetxt("mae_history_kick_policy.txt", loss_hist_mae, delimiter='\n')
np.savetxt("mse_history_kick_policy.txt", loss_hist_mse, delimiter='\n')

#Save the model
print (K.get_session().graph)
model.save('neural_kick')
