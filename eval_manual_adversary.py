from utils import eval_manual_adv
import config
from td3 import TD3
import gym 
import numpy as np 
from matplotlib import pyplot as plt 

EVAL_EPISODES = 100
STRENGTHS = [0,5,10,15,20,25,30]
RARL_mean = []
RARL_std = []
BASELINE_mean = []
BASELINE_std = []

#setup environments 
env = gym.make(config.ENV)

#get state and action dimensions 
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] 
adv_action_dim = env.adv_action_space.shape[0] 

#get protagonist limits
pro_limit = float(env.action_space.high[0])

#load rarl policy
rarl_policy = TD3(state_dim, action_dim, config.HIDDEN_LAYER_DIM, pro_limit, False, config.DISCOUNT, config.TAU, config.POLICY_NOISE, config.NOISE_CLIP, config.POLICY_FREQUENCY, config.EXPLORE_NOISE)
rarl_policy.load(config.SAVE_DIR + 'best_rarl_pro')

#load baseline policy 
baseline_policy = TD3(state_dim, action_dim, config.HIDDEN_LAYER_DIM, pro_limit, False, config.DISCOUNT, config.TAU, config.POLICY_NOISE, config.NOISE_CLIP, config.POLICY_FREQUENCY, config.EXPLORE_NOISE)
baseline_policy.load(config.SAVE_DIR + 'best_baseline')


for strength in STRENGTHS: 
    rarl_rewards = eval_manual_adv(config.ENV, config.SEED, rarl_policy, EVAL_EPISODES, config.REWARD_THRESH, config.MAX_STEPS_PER_EPISODE, strength, adv_action_dim, True)
    baseline_rewards = eval_manual_adv(config.ENV, config.SEED, baseline_policy, EVAL_EPISODES, config.REWARD_THRESH, config.MAX_STEPS_PER_EPISODE, strength, adv_action_dim, True)
    RARL_mean.append(np.mean(rarl_rewards))
    RARL_std.append(np.std(rarl_rewards))
    BASELINE_mean.append(np.mean(baseline_rewards))
    BASELINE_std.append(np.std(baseline_rewards))


plt.plot(STRENGTHS, RARL_mean, linestyle='-', color = 'g', label = 'RARL')
plt.fill_between(STRENGTHS, np.subtract(RARL_mean, RARL_std), np.add(RARL_mean, RARL_std), color='lightgreen')
plt.plot(STRENGTHS, BASELINE_mean, linestyle='-', color = 'b', label = 'Baseline')
plt.fill_between(STRENGTHS, np.subtract(BASELINE_mean, BASELINE_std), np.add(BASELINE_mean, BASELINE_std), color='lightblue', alpha=0.3)
plt.legend()
plt.xlabel('Max strength')
plt.ylabel('Reward')
plt.title(config.ENV)
plt.savefig(config.SAVE_DIR + 'manual_adv_results.png')
