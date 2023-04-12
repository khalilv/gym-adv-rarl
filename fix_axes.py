import config
import numpy as np 
from matplotlib import pyplot as plt 

with open(config.SAVE_DIR + 'results.npy', 'rb') as f:
    RARL_REWARDS = np.load(f)
    BASELINE_REWARDS = np.load(f)
    
#plot results 
x = np.arange(config.RARL_LOOPS)
RARL_mean = np.mean(RARL_REWARDS, axis=0)
RARL_std = np.std(RARL_REWARDS, axis=0)
BASELINE_mean = np.mean(BASELINE_REWARDS, axis=0)
BASELINE_std = np.std(BASELINE_REWARDS, axis=0)
plt.plot(x, RARL_mean, linestyle='-', color = 'g', label = 'RARL')
plt.fill_between(x, np.subtract(RARL_mean, RARL_std), np.add(RARL_mean, RARL_std), color='lightgreen')
plt.plot(x, BASELINE_mean, linestyle='-', color = 'b', label = 'Baseline')
plt.fill_between(x, np.subtract(BASELINE_mean, BASELINE_std), np.add(BASELINE_mean, BASELINE_std), color='lightblue', alpha=0.3)
plt.legend()
plt.xlabel('Training Iterations')
plt.ylabel('Reward')
plt.title(config.ENV)
plt.savefig(config.SAVE_DIR + 'results.png')