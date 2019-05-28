import argparse
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument('--samples', type=int, required=True)

args = parser.parse_args()

N_LAYERS = [2, 3, 4, 5, 6]
LAYER_SIZE = [16, 32, 64, 128, 256, 512]
ACTIVATION = ['tanh', 'relu']
LEARNING_RATE = [0.1, 0.01, 0.001, 0.0001]
LOSS_FUNCTION = ['mean_squared_error', 'mean_absolute_error', 'logcosh']
BATCH_SIZE = [16, 32, 64, 128, 256, 512, 1024, 2048]
EPOCHS_DROP = [10, 100, 1000]
DROP = [0.1, 0.5, 0.8, 1.0]

for i in range(args.samples):
	os.system('qsub -F \"' + str(random.choice(N_LAYERS)) + ' ' + str(random.choice(LAYER_SIZE)) + ' ' + str(random.choice(ACTIVATION)) + ' ' +
	str(random.choice(LEARNING_RATE)) + ' ' + str(random.choice(LOSS_FUNCTION)) + ' ' 
	+ str(random.choice(BATCH_SIZE)) + ' ' + str(random.choice(EPOCHS_DROP)) + ' ' + str(random.choice(DROP)) + '\" walk_supervised.sh')
	