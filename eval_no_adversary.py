from utils import test
import config
from ppo import PPO
from networks import FeedForwardNN
import gym 
import numpy as np 

EVAL_EPISODES = 100

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

#get protagonist limits
pro_limit = float(env.action_space.high[0])

#load rarl policy
rarl_policy = PPO(policy_class=FeedForwardNN, env=env, is_adv=False, is_rarl=True, **hyperparameters)
rarl_policy.load(config.SAVE_DIR + 'best_rarl_pro' + '_0') #TODO add _experiment number here for whichever is the best experiment

#load baseline policy 
baseline_policy = PPO(policy_class=FeedForwardNN, env=env, is_adv=False, is_rarl=False, **hyperparameters)
baseline_policy.load(config.SAVE_DIR + 'best_baseline' + '_0')

rarl_rewards = test(config.ENV, config.SEED, rarl_policy, EVAL_EPISODES, config.MAX_STEPS_PER_EPISODE, config.REWARD_THRESH)
baseline_rewards = test(config.ENV, config.SEED, baseline_policy, EVAL_EPISODES, config.MAX_STEPS_PER_EPISODE, config.REWARD_THRESH)

print(np.mean(rarl_rewards), np.std(rarl_rewards))
print(np.mean(baseline_rewards), np.std(baseline_rewards))