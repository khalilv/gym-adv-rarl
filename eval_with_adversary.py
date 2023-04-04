from utils import eval_with_adv
import config
from td3 import TD3
import gym 
import numpy as np 

EVAL_EPISODES = 100
ADVERSARY_STRENGTH = 10

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

#load adversary policy
adv_policy = TD3(state_dim, adv_action_dim, config.HIDDEN_LAYER_DIM, ADVERSARY_STRENGTH, True, config.DISCOUNT, config.TAU, config.POLICY_NOISE, config.NOISE_CLIP, config.POLICY_FREQUENCY, config.EXPLORE_NOISE)
adv_policy.load(config.SAVE_DIR + 'best_adv_' + str(ADVERSARY_STRENGTH))

rarl_rewards = eval_with_adv(config.ENV, config.SEED, rarl_policy, adv_policy, EVAL_EPISODES, config.REWARD_THRESH, config.MAX_STEPS_PER_EPISODE, True)
baseline_rewards = eval_with_adv(config.ENV, config.SEED, baseline_policy,adv_policy, EVAL_EPISODES, config.REWARD_THRESH, config.MAX_STEPS_PER_EPISODE, True)

print(np.mean(rarl_rewards), np.std(rarl_rewards))
print(np.mean(baseline_rewards), np.std(baseline_rewards))