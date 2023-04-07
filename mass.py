import numpy as np 
from matplotlib import pyplot as plt 
import config 

RARL_mean = [4236.85,
4513.44,
4607.2,
4693.28,
4944.91,
5064.94,
5151.13,
5110.63,
5091.37,
4992.31,
4876.67,
4705.65,
4420.97,
4087.78,
3553.84]

RARL_std = [578.65,
471.85,
620.28,
824.74,
614.14,
420.71,
87.26,
148.51,
90.55,
110.27,
115.59,
117.93,
189.98,
209.83,
199.15]

BASELINE_mean = [3940.99,
4121.47,
4254.22,
4344.71,
4424.58,
4548.97,
4580.84,
4600,
4561.62,
4465.66,
4293.55,
3893.71,
3182.29,
2936.89,
2845.49]

BASELINE_std = [98.74,
93.85,
250.52,
361,
265.43,
89.77,
91.55,
80.87,
86.52,
92.67,
121.92,
279.34,
234.94,
61.1,
49.12]

MASS = [3,
3.5,
4,
4.5,
5,
5.5,
6,
6.5,
7,
7.5,
8,
8.5,
9,
9.5,
10]


plt.plot(MASS, RARL_mean, linestyle='-', color = 'g', label = 'RARL')
plt.fill_between(MASS, np.subtract(RARL_mean, RARL_std), np.add(RARL_mean, RARL_std), color='lightgreen')
plt.plot(MASS, BASELINE_mean, linestyle='-', color = 'b', label = 'Baseline')
plt.fill_between(MASS, np.subtract(BASELINE_mean, BASELINE_std), np.add(BASELINE_mean, BASELINE_std), color='lightblue', alpha=0.3)
plt.legend()
plt.xlabel('Torso Mass')
plt.ylabel('Reward')
plt.title(config.ENV)
plt.savefig(config.SAVE_DIR + 'mass.png')