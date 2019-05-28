import numpy as np
import pandas as pd
import time
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.callbacks import LearningRateScheduler, TensorBoard

from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.python.client import device_lib
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--n_layers', type=int, required=True)
# parser.add_argument('--layer_size', type=int, required=True)
# parser.add_argument('--activation', type=str, required=True)
# parser.add_argument('--learning_rate', type=float, required=True)
# parser.add_argument('--loss_function', type=str, required=True)
# parser.add_argument('--batch_size', type=int, required=True)
# parser.add_argument('--epochs_drop', type=int, required=True)
# parser.add_argument('--drop', type=float, required=True)
args = parser.parse_args()

#Hyperparameters

start_time = time.time()

# N_LAYERS = args.n_layers
# LAYER_SIZE = args.layer_size
ACTIVATION = 'tanh'
INPUT_SIZE = 17
OUTPUT_SIZE = 14
LEARNING_RATE = 0.0001
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-08
LOSS_FUNCTION = 'mean_squared_error'
EPOCHS = 200
EPOCHS_DROP = 20
BATCH_SIZE = 128
DROP = 0.8

#Dataset processing

print("Dataset processing...")

dataset = pd.read_csv('itandroids_walk.txt', delimiter = "::", engine = 'python')
dataset = dataset.dropna(axis=0, how='any')
dataset = dataset.filter(items=['vx','vy','v_theta','curr_leftShoulderPitch','curr_rightShoulderPitch', 'curr_leftHipYawPitch', 
							'curr_leftHipRoll', 'curr_leftHipPitch', 'curr_leftKneePitch', 'curr_leftFootPitch', 'curr_leftFootRoll',
							'curr_rightHipYawPitch', 'curr_rightHipRoll', 'curr_rightHipPitch', 'curr_rightKneePitch', 'curr_rightFootPitch',
							'curr_rightFootRoll', 'desired_leftShoulderPitch', 'desired_rightShoulderPitch', 'desired_leftHipYawPitch', 
							'desired_leftHipRoll', 'desired_leftHipPitch', 'desired_leftKneePitch', 'desired_leftFootPitch', 'desired_leftFootRoll', 
							'desired_rightHipYawPitch','desired_rightHipRoll','desired_rightHipPitch','desired_rightKneePitch','desired_rightFootPitch',
							'desired_rightFootRoll'])
# dataset_1 = dataset.loc[dataset['vx'] > 0.1]
# dataset_2 = dataset.loc[dataset['vx'] < -0.1]
# dataset_3 = dataset.loc[(dataset['vx'] < 0.01) & (dataset['vx'] > -0.01)]
# dataset_3 = dataset_3.sample(1000)

# dataset_4 = dataset.loc[(dataset['vx'] < 0.1) & (dataset['vx'] > 0.01)]
# dataset_5 = dataset.loc[(dataset['vx'] > -0.1) & (dataset['vx'] < -0.01)]
# dataset_45 = pd.concat([dataset_4, dataset_5])
# dataset_45 = dataset_45.sample(10000)
# dataset = pd.concat([dataset_1, dataset_2, dataset_3, dataset_45])
train_X = dataset.values[:, :17]
train_Y = dataset.values[:, 17:]


#Neural Network Design
model =  Sequential()
# model.add(Dense(LAYER_SIZE, input_dim = INPUT_SIZE, activation = ACTIVATION))
# for i in range(N_LAYERS - 2):
#         model.add(Dense(LAYER_SIZE, input_dim = INPUT_SIZE, activation = ACTIVATION))
# model.add(Dense(OUTPUT_SIZE))
model.add(Dense(128, input_dim = INPUT_SIZE, activation = ACTIVATION))
model.add(Dense(128, input_dim = INPUT_SIZE, activation = ACTIVATION))
model.add(Dense(128, input_dim = INPUT_SIZE, activation = ACTIVATION))
model.add(Dense(128, input_dim = INPUT_SIZE, activation = ACTIVATION))
model.add(Dense(128, input_dim = INPUT_SIZE, activation = ACTIVATION))
model.add(Dense(128, activation = ACTIVATION))
model.add(Dense(128, activation = ACTIVATION))
model.add(Dense(64, activation = ACTIVATION))
model.add(Dense(OUTPUT_SIZE))

model.load_weights('walk_test_2_graph')

model.summary()
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=False)
#filename = str(args.n_layers) + '_' + str(args.layer_size) + '_' + str(args.activation) + '_' + str(args.learning_rate) + '_' + str(args.loss_function) + '_' + str(args.batch_size) + '_' + str(args.epochs_drop) + '_' + str(args.drop)
filename = 'walk_test_final'

#Training Procedure
def step_decay(epoch):
   initial_lrate = LEARNING_RATE
   drop = DROP
   epochs_drop = float(EPOCHS_DROP)
   lrate = initial_lrate * np.power(drop,  
           np.floor((1+epoch)/epochs_drop))
   if epoch % 10 == 0:
   		model.save(filename + "_graph_" + str(epoch))
   return lrate

lrate = LearningRateScheduler(step_decay)

tbCallBack = TensorBoard(log_dir='./' + filename, histogram_freq=0, write_graph=True, write_images=True)

adam = Adam(lr=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)
model.compile (loss = LOSS_FUNCTION, optimizer = adam, metrics = ['mse', 'mae'])
history_callback = model.fit (train_X, train_Y, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = 2, callbacks=[lrate, tbCallBack], validation_split=0.1)
loss_hist_mse = np.array(history_callback.history["mean_squared_error"])
loss_hist_mae = np.array(history_callback.history["mean_absolute_error"])
val_loss_hist_mae = np.array(history_callback.history["mean_absolute_error"])


#Saving Metrics History for plot
np.savetxt('mae_' +  filename + '.txt', loss_hist_mae, delimiter='\n')
np.savetxt('val_mae_' + filename + '.txt', val_loss_hist_mae, delimiter='\n')

#Save the model
print (K.get_session().graph)
model.save(filename + "_graph")
elapsed_time = time.time() - start_time
with open('elapsed_time_' + filename + '.txt','w') as f: 
	f.write(str(elapsed_time))