import numpy as np
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats


import pandas as pd

dataset = pd.read_csv('samples.csv', engine = 'python').as_matrix()


bs_mean_value = []
bs_mean_ub = []
bs_mean_lb = []

for step in dataset:
	bs_mean_step = bs.bootstrap(step, stat_func=bs_stats.mean, alpha=0.05)
	bs_std_step = bs.bootstrap(step, stat_func=bs_stats.std, alpha=0.05)
	bs_mean_value.append(bs_mean_step.value)
	bs_mean_ub.append(bs_mean_step.upper_bound)
	bs_mean_lb.append(bs_mean_step.lower_bound)


df = pd.DataFrame(np.vstack((bs_mean_value, bs_mean_lb, bs_mean_ub)))
df.T.to_csv("bootstrap_ci.csv")

