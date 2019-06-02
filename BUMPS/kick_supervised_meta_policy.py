import numpy as np
import pandas as pd

from keras.layers import Dense
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint


from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.python.client import device_lib

from sklearn.preprocessing import StandardScaler

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_layers', type=int, required=True)
parser.add_argument('--layer_size', type=int, required=True)
#parser.add_argument('--epochs_drop', type=int, required=True)
args = parser.parse_args()


#Hyperparameters

N_LAYERS = args.n_layers
LAYER_SIZE = args.layer_size
ACTIVATION = 'tanh'
INPUT_SIZE = 2
OUTPUT_SIZE = 23
LEARNING_RATE = 3e-4
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-08
LOSS_FUNCTION = 'mean_absolute_error'
EPOCHS = 150000
EPOCHS_DROP = 5000 #args.epochs_drop
BATCH_SIZE = 128
DROP = 0.8
filename = str(N_LAYERS) + "_" + str(LAYER_SIZE) + '_neural_kick_meta_policy'
print(filename)
#Dataset processing

dataset = pd.read_csv('meta-policy-dataset.txt', delimiter = "::", engine = 'python')
dataset = dataset.dropna(axis=0, how='any')
train_X = dataset.values[:, 0:2]
train_Y = dataset.values[:, 2:25]
print(train_X)


#Neural Network Design

model =  Sequential()
model.add(Dense(LAYER_SIZE, input_dim = INPUT_SIZE, activation = ACTIVATION))
for i in range(N_LAYERS - 2):
        model.add(Dense(LAYER_SIZE, activation = ACTIVATION))
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
tbCallBack = TensorBoard(log_dir='./meta_policy_tb/current/' + filename, histogram_freq=0, write_graph=True, write_images=True)
file_path="weights_base.best"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')



adam = Adam(lr=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)
model.compile (loss = LOSS_FUNCTION, optimizer = adam, metrics = ['mse', 'mae'])
history_callback = model.fit (train_X, train_Y, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = 2, callbacks=[lrate, tbCallBack])
model.save(filename)

