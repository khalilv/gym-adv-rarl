from utils import eval_manual_adv
import config
from ppo import PPO
from networks import FeedForwardNN
import gym 
import numpy as np 
from matplotlib import pyplot as plt 

EVAL_EPISODES = 100
STRENGTHS = [0,5,10,15,20,25,30,35,40,45,50]
RARL_mean = []
RARL_std = []
BASELINE_mean = []
BASELINE_std = []
hyperparameters = {
            'timesteps_per_batch': config.TIMESTEPS_PER_BATCH, 
            'max_timesteps_per_episode': config.MAX_STEPS_PER_EPISODE, 
            'gamma': config.DISCOUNT, 
            'n_updates_per_iteration': config.N_UPDATES_PER_ITERATION,
            'lr': config.LR, 
            'clip': config.POLICY_CLIP,
            'render': False,
            'render_every_i': 10
            }


#setup environments 
env = gym.make(config.ENV)

#get state and action dimensions 
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] 
adv_action_dim = env.adv_action_space.shape[0] 

#get protagonist limits
pro_limit = float(env.action_space.high[0])

#load rarl policy
rarl_policy = PPO(policy_class=FeedForwardNN, env=env, is_adv=False, is_rarl=True, **hyperparameters)
rarl_policy.load(config.SAVE_DIR + 'best_rarl_pro' + '_0') #TODO make sure to change to the best experiment number 

#load baseline policy 
baseline_policy = PPO(policy_class=FeedForwardNN, env=env, is_adv=False, is_rarl=False, **hyperparameters)
baseline_policy.load(config.SAVE_DIR + 'best_baseline' + '_0') #TODO make sure to change to the best experiment number 


for strength in STRENGTHS: 
    rarl_rewards = eval_manual_adv(config.ENV, config.SEED, rarl_policy, EVAL_EPISODES, config.REWARD_THRESH, config.MAX_STEPS_PER_EPISODE, strength, adv_action_dim, False)
    baseline_rewards = eval_manual_adv(config.ENV, config.SEED, baseline_policy, EVAL_EPISODES, config.REWARD_THRESH, config.MAX_STEPS_PER_EPISODE, strength, adv_action_dim, False)
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
