import numpy as np 
from ActionWrapper import ProAdvAction

class Runner():
    def __init__(self, env, pro_agent, adv_agent):
        self.env = env
        self.pro_agent = pro_agent
        self.pro_agent.set_runner(self) #link with runner
        self.adv_agent = adv_agent
        self.adv_agent.set_runner(self) #link with runner        
        self.obs = env.reset()
        self.done = False
    
    #When called, generates an action from the protagonist and adversary, and feeds it to the gym env. Returns the generated action and transition.
    def next_step(self, adv_turn): 
        if adv_turn:
            adv_action, adv_log_prob = (self.adv_agent.get_action(np.array(self.obs)))
            #    + np.random.normal(0, self.adv_agent.limit * self.adv_agent.expl_noise, size=self.adv_agent.action_dim)
            #).clip(-self.adv_agent.limit, self.adv_agent.limit)
            pro_action, pro_log_prob = self.pro_agent.get_action(np.array(self.obs))
        else:
            adv_action, adv_log_prob = self.adv_agent.get_action(np.array(self.obs))
            pro_action, pro_log_prob = (self.pro_agent.get_action(np.array(self.obs)))
            #    + np.random.normal(0, self.pro_agent.limit * self.pro_agent.expl_noise, size=self.pro_agent.action_dim)
            #).clip(-self.pro_agent.limit, self.pro_agent.limit)

        action = ProAdvAction(pro=pro_action, adv=adv_action) #mix actions together
        new_obs, reward, done, _ = self.env.step(action) #generate new observation from both actions
        self.obs = new_obs #point to new observation

        if adv_turn:
            return adv_action, adv_log_prob, -reward, done, _
        else:
            return pro_action, pro_log_prob, reward, done, _
    
    def reset(self):
        self.obs = self.env.reset()

    def rollout(self, timesteps_per_batch, is_adv):
        """
        Too many transformers references, I'm sorry. This is where we collect the batch of data
        from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
        of data each time we iterate the actor/critic networks.
        Parameters:
            None
        Return:
            batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
            batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
            batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
            batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
            batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
        """
        # Batch data. For more details, check function header.
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = [] 

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []

        t = 0 # Keeps track of how many timesteps we've run so far this batch


        #QUERY THE REPLAY BUFFER HERE, FOR X AMOUNT OF timesteps per batch. 
        #
        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < timesteps_per_batch:
            ep_rews = [] # rewards collected per episode

            # Reset the environment. sNote that obs is short for observation. 
            self.obs = self.env.reset()
            done = False

            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.pro_agent.max_timesteps_per_episode): 
                # If render is specified, render the environment
                #if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
                    #self.env.render()

                t += 1 # Increment timesteps ran this batch so far

                # Track observations in this batch
                batch_obs.append(self.obs)

                # Calculate action and make a step in the env. 
                # Note that rew is short for reward.

                action, log_prob, rew, done, _ = self.next_step(is_adv)

                # Track recent reward, action, and action log probability
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                # If the environment tells us the episode is terminated, break
                if done:
                    break

            # Track episodic lengths and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)



        return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens