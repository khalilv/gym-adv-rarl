
from random import randint
import sys
import numpy as np
from ActionWrapper import ProAdvAction
import gym
import torch
from td3 import TD3
from replay_buffer import ReplayBuffer
from runner import Runner
import config
from matplotlib import pyplot as plt

def observe(env, replay_buffer, observation_steps, max_steps_per_episode):
    time_steps = 0
    episode_steps = 0
    obs = env.reset()
    done = False

    while time_steps < observation_steps:
        action = env.sample_action()
        new_obs, reward, done, _ = env.step(action)
        replay_buffer.add(obs, action.pro, action.adv, reward, new_obs, done)
        obs = new_obs
        time_steps += 1
        episode_steps += 1
        if done or episode_steps >= max_steps_per_episode:
            obs = env.reset()
            episode_steps = 0
        print("\rPopulating Buffer {}/{}.".format(time_steps, observation_steps), end="")
        sys.stdout.flush()

def train(agent, N, replay_buffer, runner, batch_size, reward_threshold, max_steps_per_episode):
    episode_num = 0
    episode_reward = 0
    episode_steps = 0
    done = False
    rewards = []
    runner.reset()
    for i in range(N):
        if done or episode_reward > reward_threshold or episode_steps >= max_steps_per_episode:
            rewards.append(episode_reward)
            print("\rUpdates: {:d} Episode Num: {:d} Reward: {:f} Avg Reward: {:f}\n".format(
                i, episode_num, episode_reward, np.mean(rewards[-50:])), end="")
            sys.stdout.flush()
            episode_reward = 0
            episode_steps = 0
            episode_num += 1
            runner.reset()
        reward, done = runner.next_step(agent.is_adv)
        episode_reward += reward
        episode_steps += 1
        agent.train(replay_buffer, batch_size)

def eval(policy, episodes, reward_threshold, max_steps_per_episode, render=False):
    eval_env = gym.make(config.ENV)
    eval_env.seed(config.SEED)
    obs = eval_env.reset()
    rewards = []
    episode_reward = 0
    episode_steps = 0
    ep = 0
    render_at = randint(0, episodes) 
    while ep < episodes:
        action = policy.select_action(np.array(obs))
        new_obs, reward, done, _ = eval_env.step(action)
        if ep == render_at and render:
            eval_env.render()
        episode_reward += reward
        episode_steps += 1
        obs = new_obs
        if done or episode_reward > reward_threshold or episode_steps >= max_steps_per_episode:
            rewards.append(episode_reward)
            episode_reward = 0
            episode_steps = 0
            if ep == render_at and render:
                eval_env.render(close=True)
            obs = eval_env.reset()
            ep += 1
    
    print("\n---------------------------------------")
    print(f"Evaluation over {episodes} episodes: {np.mean(rewards):.3f}")
    print("---------------------------------------\n")
    return np.mean(rewards)

best_observed_rarl = -1
best_observed_baseline = -1 
RARL_REWARDS = []
BASELINE_REWARDS = []

for ex in range(config.NUM_EXPERIMENTS):
    
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

    rarl_rewards = []
    baseline_rewards = []

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
        rarl_reward = eval(adv_env_pro_policy,config.EVAL_EPISODES, config.REWARD_THRESH, config.MAX_STEPS_PER_EPISODE)
        
        #evaluate baseline
        baseline_reward = eval(base_env_pro_policy, config.EVAL_EPISODES, config.REWARD_THRESH, config.MAX_STEPS_PER_EPISODE)

        #store results
        rarl_rewards.append(rarl_reward)
        baseline_rewards.append(baseline_reward)

        if rarl_reward >= best_observed_rarl:
            adv_env_pro_policy.save(config.SAVE_DIR + 'best_rarl_pro')
            adv_env_adv_policy.save(config.SAVE_DIR + 'best_rarl_adv')
            best_observed_rarl = rarl_reward

        if baseline_reward >= best_observed_baseline:
            base_env_pro_policy.save(config.SAVE_DIR + 'best_baseline')
            best_observed_baseline = baseline_reward

    RARL_REWARDS.append(rarl_rewards)
    BASELINE_REWARDS.append(baseline_rewards)

with open(config.SAVE_DIR + 'results.npy', 'wb') as f:
    np.save(f, np.array(RARL_REWARDS))
    np.save(f, np.array(BASELINE_REWARDS))


plt.errorbar(np.arange(config.RARL_LOOPS), np.mean(RARL_REWARDS, axis=0), np.std(RARL_REWARDS, axis=0), linestyle='-', color = 'g', ecolor = 'lightgreen', label = 'RARL')
plt.errorbar(np.arange(config.RARL_LOOPS), np.mean(BASELINE_REWARDS, axis=0), np.std(BASELINE_REWARDS, axis=0), linestyle='-', color = 'b', ecolor = 'lightblue', label = 'Baseline')
plt.legend()
plt.savefig(config.SAVE_DIR + 'results.png')
