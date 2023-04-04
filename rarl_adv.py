import numpy as np
import gym
from td3 import TD3
from replay_buffer import ReplayBuffer
from runner import Runner
import config
from matplotlib import pyplot as plt
from utils import train, eval_with_adv, observe

RARL_REWARDS = []
ADVERSARY_STENGTHS = [5,10,20,30,40]

for strength in ADVERSARY_STENGTHS:
    best_observed_rarl = config.REWARD_THRESH

    #setup environments 
    adv_env = gym.make(config.ENV)
    adv_env.update_adversary(strength)

    #get state and action dimensions 
    state_dim = adv_env.observation_space.shape[0]
    pro_action_dim = adv_env.action_space.shape[0] 
    adv_action_dim = adv_env.adv_action_space.shape[0] 

    #get protagonist and antagonist limits
    pro_limit = float(adv_env.action_space.high[0])
    adv_env_adv_limit = float(adv_env.adv_action_space.high[0])

    #load policies 
    adv_env_pro_policy = TD3(state_dim, pro_action_dim, config.HIDDEN_LAYER_DIM, pro_limit, False, config.DISCOUNT, config.TAU, config.POLICY_NOISE, config.NOISE_CLIP, config.POLICY_FREQUENCY, config.EXPLORE_NOISE)
    adv_env_pro_policy.load(config.SAVE_DIR + 'best_rarl_pro')

    baseline_env_pro_policy = TD3(state_dim, pro_action_dim, config.HIDDEN_LAYER_DIM, pro_limit, False, config.DISCOUNT, config.TAU, config.POLICY_NOISE, config.NOISE_CLIP, config.POLICY_FREQUENCY, config.EXPLORE_NOISE)
    baseline_env_pro_policy.load(config.SAVE_DIR + 'best_baseline')

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
        print("\nEVALUATING RARL POLICY WITH ADVERSARY: ", i, "\n")
        rarl_reward = eval_with_adv(config.ENV, config.SEED, adv_env_pro_policy, adv_env_adv_policy, config.EVAL_EPISODES, config.REWARD_THRESH, config.MAX_STEPS_PER_EPISODE, False)
        
        #evaluate baseline - dont store learning curve results (just for verification)
        print("\nEVALUATING BASELINE POLICY WITH ADVERSARY: ", i, "\n")
        eval_with_adv(config.ENV, config.SEED, baseline_env_pro_policy, adv_env_adv_policy, config.EVAL_EPISODES, config.REWARD_THRESH, config.MAX_STEPS_PER_EPISODE, False)
        
        #store results
        rarl_rewards.append(np.mean(rarl_reward))

        #save best adversary for each strength
        if np.mean(rarl_reward) <= best_observed_rarl:
            adv_env_adv_policy.save(config.SAVE_DIR + 'best_adv_' + str(strength))
            best_observed_rarl = np.mean(rarl_reward)

    RARL_REWARDS.append(rarl_rewards)

with open(config.SAVE_DIR + 'adv_results_' + str(strength) + '.npy', 'wb') as f:
    np.save(f, np.array(RARL_REWARDS))

#plot learning curves. Should be decreasing over time as the adversary learns good actions to take 
for i in range(len(ADVERSARY_STENGTHS)):
    plt.plot(np.arange(config.RARL_LOOPS), RARL_REWARDS[i,:], linestyle='-', label = str(ADVERSARY_STENGTHS[i]))
plt.legend()
plt.title(config.ENV)
plt.savefig(config.SAVE_DIR + 'adv_results.png')
