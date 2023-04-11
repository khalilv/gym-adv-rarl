import numpy as np 
from matplotlib import pyplot as plt 
import config 

#add data to plot here
RARL_mean = []
RARL_std = []
BASELINE_mean = []
BASELINE_std = []
MASS = []


plt.plot(MASS, RARL_mean, linestyle='-', color = 'g', label = 'RARL')
plt.fill_between(MASS, np.subtract(RARL_mean, RARL_std), np.add(RARL_mean, RARL_std), color='lightgreen')
plt.plot(MASS, BASELINE_mean, linestyle='-', color = 'b', label = 'Baseline')
plt.fill_between(MASS, np.subtract(BASELINE_mean, BASELINE_std), np.add(BASELINE_mean, BASELINE_std), color='lightblue', alpha=0.3)
plt.legend()
plt.xlabel('Torso Mass')
plt.ylabel('Reward')
plt.title(config.ENV)
plt.savefig(config.SAVE_DIR + 'mass.png')