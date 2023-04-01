import numpy as np
import gym
from td3 import TD3
from replay_buffer import ReplayBuffer
from runner import Runner
import config
from matplotlib import pyplot as plt
from utils import train, eval_with_adv, observe

best_observed_rarl = config.REWARD_THRESH
RARL_REWARDS = []

for ex in range(config.NUM_EXPERIMENTS):
    
    #setup environments 
    adv_env = gym.make(config.ENV)
    adv_env.update_adversary(config.MAX_ADVERSARY_STRENGTH)

    #get state and action dimensions 
    state_dim = adv_env.observation_space.shape[0]
    pro_action_dim = adv_env.action_space.shape[0] 
    adv_action_dim = adv_env.adv_action_space.shape[0] 

    #get protagonist and antagonist limits
    pro_limit = float(adv_env.action_space.high[0])
    adv_env_adv_limit = float(adv_env.adv_action_space.high[0])

    #create policies 
    adv_env_pro_policy = TD3(state_dim, pro_action_dim, config.HIDDEN_LAYER_DIM, pro_limit, False, config.DISCOUNT, config.TAU, config.POLICY_NOISE, config.NOISE_CLIP, config.POLICY_FREQUENCY, config.EXPLORE_NOISE)
    adv_env_pro_policy.load(config.SAVE_DIR + 'best_rarl_pro')

    adv_env_adv_policy = TD3(state_dim, adv_action_dim, config.HIDDEN_LAYER_DIM, adv_env_adv_limit, True, config.DISCOUNT, config.TAU, config.POLICY_NOISE, config.NOISE_CLIP, config.POLICY_FREQUENCY, config.EXPLORE_NOISE)
    
    #create replay buffers 
    adv_env_replay_buffer = ReplayBuffer(config.MAX_REPLAY_BUFFER_SIZE)

    #create runners
    adv_env_runner = Runner(adv_env, adv_env_pro_policy, adv_env_adv_policy, adv_env_replay_buffer)

    #fill in replay buffers with random actions
    observe(adv_env, adv_env_replay_buffer, config.INITIAL_OBSERVATION_STEPS, config.MAX_STEPS_PER_EPISODE)

    rarl_rewards = []

    for i in range(config.RARL_LOOPS):
                
        #train adversary
        print("\nTraining RARL adversary: ", i, "\n")
        train(adv_env_adv_policy, config.N_TRAINING, adv_env_replay_buffer, adv_env_runner, config.BATCH_SIZE, config.REWARD_THRESH, config.MAX_STEPS_PER_EPISODE)

              
        #evaluate RARL
        rarl_reward = eval_with_adv(config.ENV, config.SEED, adv_env_pro_policy, adv_env_adv_policy, config.EVAL_EPISODES, config.REWARD_THRESH, config.MAX_STEPS_PER_EPISODE)
        
        #store results
        rarl_rewards.append(np.mean(rarl_reward))

        if np.mean(rarl_reward) <= best_observed_rarl:
            adv_env_adv_policy.save(config.SAVE_DIR + 'best_adv')
            best_observed_rarl = np.mean(rarl_reward)

    RARL_REWARDS.append(rarl_rewards)

with open(config.SAVE_DIR + 'adv_results.npy', 'wb') as f:
    np.save(f, np.array(RARL_REWARDS))

plt.errorbar(np.arange(config.RARL_LOOPS), np.mean(RARL_REWARDS, axis=0), np.std(RARL_REWARDS, axis=0), linestyle='-', color = 'g', ecolor = 'lightgreen', label = 'RARL')
plt.legend()
plt.savefig(config.SAVE_DIR + 'adv_results.png')
