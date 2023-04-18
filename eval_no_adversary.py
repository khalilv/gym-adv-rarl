from utils import eval
import config
from ppo_old import PPO
import gym 
import numpy as np 

EVAL_EPISODES = 100

#setup environments 
env = gym.make(config.ENV)

#get state and action dimensions 
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] 

#get protagonist limits
pro_limit = float(env.action_space.high[0])

#load rarl policy
rarl_policy = PPO(state_dim, action_dim, config.HIDDEN_LAYER_DIM, pro_limit, False, config.DISCOUNT, config.GAE_LAMBDA, config.POLICY_CLIP, config.BATCH_SIZE, config.N, config.N_EPOCHS, config.EXPLORE_NOISE)
rarl_policy.load(config.SAVE_DIR + 'best_rarl_pro')

#load baseline policy 
baseline_policy = PPO(state_dim, action_dim, config.HIDDEN_LAYER_DIM, pro_limit, False, config.DISCOUNT, config.GAE_LAMBDA, config.POLICY_CLIP, config.BATCH_SIZE, config.N, config.N_EPOCHS, config.EXPLORE_NOISE)
baseline_policy.load(config.SAVE_DIR + 'best_baseline')

rarl_rewards = eval(config.ENV, config.SEED, rarl_policy, EVAL_EPISODES, config.REWARD_THRESH, config.MAX_STEPS_PER_EPISODE, False)
baseline_rewards = eval(config.ENV, config.SEED, baseline_policy, EVAL_EPISODES, config.REWARD_THRESH, config.MAX_STEPS_PER_EPISODE, False)

print(np.mean(rarl_rewards), np.std(rarl_rewards))
print(np.mean(baseline_rewards), np.std(baseline_rewards))