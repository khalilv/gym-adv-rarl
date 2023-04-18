"""
	This file is used only to evaluate our trained policy/actor after
	training in main.py with ppo.py. I wrote this file to demonstrate
	that our trained policy exists independently of our learning algorithm,
	which resides in ppo.py. Thus, we can test our trained policy without 
	relying on ppo.py.
"""

import numpy as np


def _log_summary(ep_len, ep_ret, eval_episodes):
		"""
			Print to stdout what we've logged so far in the most recent episode.
			Parameters:
				None
			Return:
				None
		"""
		# Round decimal places for more aesthetic logging messages
		ep_len = str(round(ep_len, 2))
		ep_ret = str(round(ep_ret, 2))

		# Print logging statements
		print(flush=True)
		print(f"--------- Evaluation Results for {eval_episodes} Episodes ----------", flush=True)
		print(f"Mean Episodic Length: {ep_len}", flush=True)
		print(f"Mean Episodic Return: {ep_ret}", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

def rollout(policy, env, eval_episodes, max_timesteps, rew_thresh, render):
	"""
		Returns a generator to roll out each episode given a trained policy and
		environment to test on. 
		Parameters:
			policy - The trained policy to test
			env - The environment to evaluate the policy on
			render - Specifies whether to render or not
		
		Return:
			A generator object rollout, or iterable, which will return the latest
			episodic length and return on each iteration of the generator.
		Note:
			If you're unfamiliar with Python generators, check this out:
				https://wiki.python.org/moin/Generators
			If you're unfamiliar with Python "yield", check this out:
				https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
	"""
	# Rollout until specified
	episodes = 0
	ret_arr = []
	len_arr = []
	while episodes < eval_episodes:
	
		obs = env.reset()
		done = False

		# number of timesteps so far
		t = 0

		# Logging data
		ep_len = 0            # episodic length
		ep_ret = 0            # episodic return

		while not done and t < max_timesteps and ep_ret < rew_thresh:
			t += 1

			# Render environment if specified, off by default
			#if render:
				#env.render()

			# Query deterministic action from policy and run it
			action, _ = policy.get_action(obs)
			obs, rew, done, _ = env.step(action)

			# Sum all episodic rewards as we go along
			ep_ret += rew
			
		# Track episodic length
		ep_len = t
		episodes += 1
		ret_arr.append(ep_ret)
		len_arr.append(ep_len)
		# returns episodic length and return in this iteration
	return ret_arr, len_arr


def eval_policy(policy, env, eval_episodes, max_timesteps, rew_thresh, render=False):

	# Rollout with the policy and environment, and log each episode's data
	ret_arr, len_arr = rollout(policy, env, eval_episodes, max_timesteps, rew_thresh, render)
	_log_summary(ep_len=np.mean(len_arr), ep_ret=np.mean(ret_arr), eval_episodes=eval_episodes)
	return ret_arr