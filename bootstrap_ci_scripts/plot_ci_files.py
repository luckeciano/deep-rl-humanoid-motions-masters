from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--files', nargs='+')
parser.add_argument('--figure', nargs='+')
parser.add_argument('--title', nargs='+')

args = parser.parse_args()
figure_name = args.figure[0]
title = args.title[0]

fig, ax = plt.subplots()

for file in args.files:
	df = pd.read_csv(file, engine = 'python').as_matrix()[2000:]
	bs_mean_value = df.T[1]
	bs_mean_lb = df.T[2]
	bs_mean_ub = df.T[3]

	t = np.arange(len(bs_mean_value))
	# Create the plot object


	# Plot the data, set the linewidth, color and transparency of the
	# line, provide a label for the legend
	ax.plot(t, bs_mean_value, lw = 1, label = file[:-4])
	# Shade the confidence interval
	ax.fill_between(t, bs_mean_lb, bs_mean_ub,  alpha = 0.4)
	# Label the axes and provide a title
	ax.set_title(title)
	ax.set_xlabel("Kick Episode")
	ax.set_ylabel("Error")

	# Display legend
	ax.legend(loc = 'best')

ax.grid()
fig.savefig(figure_name)