import sys
from random import randint
import numpy as np 
import gym 
from ActionWrapper import ProAdvAction
import torch
from eval_policy import eval_policy
from networks import FeedForwardNN

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

# def train(agent, N, runner, reward_threshold, max_steps_per_episode):
#     episode_num = 0
#     episode_reward = 0
#     episode_steps = 0
#     done = False
#     rewards = []
#     runner.reset()
#     for i in range(N):
#         if done or episode_reward > reward_threshold or episode_steps >= max_steps_per_episode:
#             rewards.append(episode_reward)
#             print("\rUpdates: {:d} Episode Num: {:d} Reward: {:f} Avg Reward: {:f}\n".format(
#                 i, episode_num, episode_reward, np.mean(rewards[-50:])), end="")
#             sys.stdout.flush()
#             episode_reward = 0
#             episode_steps = 0
#             episode_num += 1
#             runner.reset()

#         reward, done = runner.next_step(agent.is_adv)

#         episode_reward += reward
#         episode_steps += 1
#         agent.train()

def train(model, training_timesteps, actor_model, critic_model):

    print(f"Training", flush=True)
    # Tries to load in an existing actor/critic model to continue training on
    if actor_model != '' and critic_model != '':
        print(f"Loading in {actor_model} and {critic_model}...", flush=True)
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print(f"Successfully loaded.", flush=True)

    elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
        print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
        sys.exit(0)
                
    else:
        print(f"Training from scratch.", flush=True)

	# Train the PPO model with a specified total timesteps
	# NOTE: You can change the total timesteps here, I put a big number just because
	# you can kill the process whenever you feel like PPO is converging
    model.learn(total_timesteps=training_timesteps)

def test(env, seed, actor_model, eval_episodes, max_timesteps, rew_thresh):
    """
    Tests the model.
    Parameters:
        env - the environment to test the policy on
        actor_model - the actor model to load in
    Return:
        None
    """
    test_env= gym.make(env)
    test_env.seed(seed)

    print(f"Testing {actor_model}", flush=True)

    
                
    #Extract out dimensions of observation and action spaces
    obs_dim = test_env.observation_space.shape[0]
    act_dim = test_env.action_space.shape[0]


    policy = actor_model
        
    # Evaluate our policy with a separate module, eval_policy, to demonstrate
    # that once we are done training the model/policy with ppo.py, we no longer need
    # ppo.py since it only contains the training algorithm. The model/policy itself exists
    # independently as a binary file that can be loaded in with torch.


    total_ret = eval_policy(policy, test_env, eval_episodes, max_timesteps, rew_thresh, render=False)

    return total_ret


def eval_manual_adv(env_id, seed, policy, episodes, reward_threshold, max_steps_per_episode, max_strength, adv_dim, render=False):
    eval_env = gym.make(env_id)
    eval_env.update_adversary(max_strength)
    print(adv_dim,"ADV_DIM")
    eval_env.seed(seed)
    np.random.seed(seed)
    obs = eval_env.reset()
    rewards = []
    episode_reward = 0
    episode_steps = 0
    ep = 0
    render_at = randint(0, episodes) 
    while ep < episodes:
        #define adv action here
        adv_action = np.zeros(adv_dim)
        
        #pendulum 
        # adv_action[0] = np.random.uniform(-max_strength, max_strength)
        # adv_action[1] = np.random.uniform(-max_strength, max_strength)

        #hopper
        adv_action[0] = np.random.uniform(-max_strength,0)
        adv_action[1] = np.random.uniform(-max_strength, max_strength)

        #halfcheetah
        # adv_action[0] = np.random.uniform(-max_strength,0)
        # adv_action[1] = np.random.uniform(-max_strength, max_strength)
        # adv_action[2] = np.random.uniform(-max_strength,0)
        # adv_action[3] = np.random.uniform(-max_strength, max_strength)
        # adv_action[4] = np.random.uniform(-max_strength,0)
        # adv_action[5] = np.random.uniform(-max_strength, max_strength)
        
        action = ProAdvAction(pro = policy.get_action(np.array(obs))[0], adv = adv_action)
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


