import numpy as np
import gym
from ppo import PPO
from networks import FeedForwardNN
from runner import Runner
import config
from matplotlib import pyplot as plt
from utils import train, test

RARL_REWARDS = np.zeros((config.NUM_EXPERIMENTS * config.EVAL_EPISODES, config.RARL_LOOPS))
BASELINE_REWARDS = np.zeros((config.NUM_EXPERIMENTS * config.EVAL_EPISODES, config.RARL_LOOPS))



for ex in range(config.NUM_EXPERIMENTS):
    
    best_observed_rarl = -1
    best_observed_baseline = -1 
    #setup environments 
    adv_env = gym.make(config.ENV)
    base_env = gym.make(config.ENV)
    adv_env.update_adversary(config.MAX_ADVERSARY_STRENGTH)
    base_env.update_adversary(0)

    #get state and action dimensions 
    state_dim = adv_env.observation_space.shape[0]
    pro_action_dim = adv_env.action_space.shape[0] 
    adv_action_dim = adv_env.adv_action_space.shape[0] 
    
    #get protagonist and antagonist limits
    pro_limit = float(adv_env.action_space.high[0])
    adv_env_adv_limit = float(adv_env.adv_action_space.high[0])
    base_env_adv_limit = float(base_env.adv_action_space.high[0]) #should be 0


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
    
    #create policies self, state_dim, action_dim, hidden_dim, limit, is_adv, gamma=0.99, gae_lambda=0.95, policy_clip=0.2, batch_size=64, N=2048, n_epochs=10
    adv_env_pro_policy = PPO(policy_class=FeedForwardNN, env=adv_env, is_adv=False, is_rarl=True, **hyperparameters)
    adv_env_adv_policy = PPO(policy_class=FeedForwardNN, env=adv_env, is_adv=True, is_rarl=False, **hyperparameters)
    base_env_pro_policy = PPO(policy_class=FeedForwardNN, env=base_env, is_adv=False, is_rarl=False, **hyperparameters)
    base_env_adv_policy = PPO(policy_class=FeedForwardNN, env=base_env, is_adv=True, is_rarl=False, **hyperparameters)

    #create runners
    adv_env_runner = Runner(adv_env, adv_env_pro_policy, adv_env_adv_policy)
    base_env_runner = Runner(base_env, base_env_pro_policy, base_env_adv_policy)


    for i in range(config.RARL_LOOPS):
        #train protagonist
        print("\nTraining RARL: ", i, "\n")
        train(adv_env_pro_policy, config.N_TRAINING, actor_model='', critic_model='') #Don't need to pass adversary, as it is linked via the runner
        
        #train adversary
        print("\nTraining RARL adversary: ", i, "\n")
        train(adv_env_adv_policy, config.N_TRAINING, actor_model='', critic_model='')

        #train baseline (zero strength adversary)
        print("\nTraining baseline: ", i, "\n")
        train(base_env_pro_policy , config.N_TRAINING, actor_model='', critic_model='')
        
        #evaluate RARL
        rarl_reward = test(config.ENV, config.SEED, adv_env_pro_policy, config.EVAL_EPISODES, config.MAX_STEPS_PER_EPISODE, config.REWARD_THRESH)
        
        #evaluate baseline
        baseline_reward = test(config.ENV, config.SEED, base_env_pro_policy, config.EVAL_EPISODES, config.MAX_STEPS_PER_EPISODE, config.REWARD_THRESH)

        #store results
        RARL_REWARDS[int(ex*config.EVAL_EPISODES):int(ex*config.EVAL_EPISODES) + config.EVAL_EPISODES, i] = rarl_reward
        BASELINE_REWARDS[int(ex*config.EVAL_EPISODES):int(ex*config.EVAL_EPISODES) + config.EVAL_EPISODES, i] = baseline_reward

        if np.mean(rarl_reward) >= best_observed_rarl:
            adv_env_pro_policy.save(config.SAVE_DIR + 'best_rarl_pro_' + str(ex))
            adv_env_adv_policy.save(config.SAVE_DIR + 'best_rarl_adv_' + str(ex))
            best_observed_rarl = np.mean(rarl_reward)

        if np.mean(baseline_reward) >= best_observed_baseline:
            base_env_pro_policy.save(config.SAVE_DIR + 'best_baseline_' + str(ex))
            best_observed_baseline = np.mean(baseline_reward)

with open(config.SAVE_DIR + 'results.npy', 'wb') as f:
   # np.save(f, np.array(RARL_REWARDS))
    np.save(f, np.array(BASELINE_REWARDS))


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
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title(config.ENV)
plt.savefig(config.SAVE_DIR + 'results.png')