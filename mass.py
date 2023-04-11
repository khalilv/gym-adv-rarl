import numpy as np 
from matplotlib import pyplot as plt 
import config 

RARL_mean = [3199.92,
3263.4,
3316.82,
3373.52,
3429.43,
3468.66,
3467.86,
3479.67,
3326.58,
2283.14,
1202.65, 1030.92, 1017.32]

RARL_std = [12.9,
12.1,
6.34,
5.78,
4.64,
12.14,
11.83,
22.09,
577.9,
1078.2,
135.94, 24.33, 58.75]

BASELINE_mean = [3139.73,
3182.78,
3211.77,
3238.03,
3242.53,
3275.9,
3293.64,
2812.1,
1489.24,
1113.83,
1022.74, 924.9, 908.66]

BASELINE_std = [24.09,
19.03,
14.44,
13.48,
153.86,
5.8,
7.67,
852.38,
803.62,
482.37,
417.97, 30.64, 38.51]

MASS = [2.5,
2.75,
3,
3.25,
3.5,
3.75,
4,
4.25,
4.5,
4.75,
5,
5.25,
5.5]


plt.plot(MASS, RARL_mean, linestyle='-', color = 'g', label = 'RARL')
plt.fill_between(MASS, np.subtract(RARL_mean, RARL_std), np.add(RARL_mean, RARL_std), color='lightgreen')
plt.plot(MASS, BASELINE_mean, linestyle='-', color = 'b', label = 'Baseline')
plt.fill_between(MASS, np.subtract(BASELINE_mean, BASELINE_std), np.add(BASELINE_mean, BASELINE_std), color='lightblue', alpha=0.3)
plt.legend()
plt.xlabel('Torso Mass')
plt.ylabel('Reward')
plt.title(config.ENV)
plt.savefig(config.SAVE_DIR + 'mass.png')