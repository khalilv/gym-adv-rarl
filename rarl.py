import numpy as np
import gym
from td3 import TD3
from replay_buffer import ReplayBuffer
from runner import Runner
import config
from matplotlib import pyplot as plt
from utils import train, eval, observe

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

    #create policies 
    adv_env_pro_policy = TD3(state_dim, pro_action_dim, config.HIDDEN_LAYER_DIM, pro_limit, False, config.DISCOUNT, config.TAU, config.POLICY_NOISE, config.NOISE_CLIP, config.POLICY_FREQUENCY, config.EXPLORE_NOISE)
    adv_env_adv_policy = TD3(state_dim, adv_action_dim, config.HIDDEN_LAYER_DIM, adv_env_adv_limit, True, config.DISCOUNT, config.TAU, config.POLICY_NOISE, config.NOISE_CLIP, config.POLICY_FREQUENCY, config.EXPLORE_NOISE)
    base_env_pro_policy = TD3(state_dim, pro_action_dim, config.HIDDEN_LAYER_DIM, pro_limit, False, config.DISCOUNT, config.TAU, config.POLICY_NOISE, config.NOISE_CLIP, config.POLICY_FREQUENCY, config.EXPLORE_NOISE)
    base_env_adv_policy = TD3(state_dim, adv_action_dim, config.HIDDEN_LAYER_DIM, base_env_adv_limit, True, config.DISCOUNT, config.TAU, config.POLICY_NOISE, config.NOISE_CLIP, config.POLICY_FREQUENCY, config.EXPLORE_NOISE)

    #create replay buffers 
    adv_env_replay_buffer = ReplayBuffer(config.MAX_REPLAY_BUFFER_SIZE)
    base_env_replay_buffer = ReplayBuffer(config.MAX_REPLAY_BUFFER_SIZE)

    #create runners
    adv_env_runner = Runner(adv_env, adv_env_pro_policy, adv_env_adv_policy, adv_env_replay_buffer)
    base_env_runner = Runner(base_env, base_env_pro_policy, base_env_adv_policy, base_env_replay_buffer)

    #fill in replay buffers with random actions
    observe(base_env, base_env_replay_buffer, config.INITIAL_OBSERVATION_STEPS, config.MAX_STEPS_PER_EPISODE)
    observe(adv_env, adv_env_replay_buffer, config.INITIAL_OBSERVATION_STEPS, config.MAX_STEPS_PER_EPISODE)

    for i in range(config.RARL_LOOPS):
        #train protagonist
        print("\nTraining RARL: ", i, "\n")
        train(adv_env_pro_policy, config.N_TRAINING, adv_env_replay_buffer, adv_env_runner, config.BATCH_SIZE, config.REWARD_THRESH, config.MAX_STEPS_PER_EPISODE)
        
        #train adversary
        print("\nTraining RARL adversary: ", i, "\n")
        train(adv_env_adv_policy, config.N_TRAINING, adv_env_replay_buffer, adv_env_runner, config.BATCH_SIZE, config.REWARD_THRESH, config.MAX_STEPS_PER_EPISODE)

        #train baseline (zero strength adversary)
        print("\nTraining baseline: ", i, "\n")
        train(base_env_pro_policy, config.N_TRAINING, base_env_replay_buffer, base_env_runner, config.BATCH_SIZE, config.REWARD_THRESH, config.MAX_STEPS_PER_EPISODE)
        
        #evaluate RARL
        rarl_reward = eval(config.ENV, config.SEED, adv_env_pro_policy,config.EVAL_EPISODES, config.REWARD_THRESH, config.MAX_STEPS_PER_EPISODE)
        
        #evaluate baseline
        baseline_reward = eval(config.ENV, config.SEED, base_env_pro_policy, config.EVAL_EPISODES, config.REWARD_THRESH, config.MAX_STEPS_PER_EPISODE)

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
    np.save(f, np.array(RARL_REWARDS))
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