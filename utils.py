import sys
from random import randint
import numpy as np 
import gym 

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

def eval(env_id, seed, policy, episodes, reward_threshold, max_steps_per_episode, render=False):
    eval_env = gym.make(env_id)
    eval_env.seed(seed)
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
    return rewards


