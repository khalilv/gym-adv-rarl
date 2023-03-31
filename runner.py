import numpy as np 
from ActionWrapper import ProAdvAction

class Runner():
    def __init__(self, env, pro_agent, adv_agent, replay_buffer):
        self.env = env
        self.pro_agent = pro_agent
        self.adv_agent = adv_agent
        self.replay_buffer = replay_buffer
        self.obs = env.reset()
        self.done = False
    
    def next_step(self, adv_turn):
        if adv_turn:
            adv_action = (self.adv_agent.select_action(np.array(self.obs))
                + np.random.normal(0, self.adv_agent.limit * self.adv_agent.expl_noise, size=self.adv_agent.action_dim)
            ).clip(-self.adv_agent.limit, self.adv_agent.limit)
            pro_action = self.pro_agent.select_action(np.array(self.obs))
        else:
            adv_action = self.adv_agent.select_action(np.array(self.obs))
            pro_action = (self.pro_agent.select_action(np.array(self.obs))
                + np.random.normal(0, self.pro_agent.limit * self.pro_agent.expl_noise, size=self.pro_agent.action_dim)
            ).clip(-self.pro_agent.limit, self.pro_agent.limit)
        action = ProAdvAction(pro=pro_action, adv=adv_action)
        new_obs, reward, done, _ = self.env.step(action)
        self.replay_buffer.add(self.obs, action.pro, action.adv, reward, new_obs, done)
        self.obs = new_obs
        if done:
            self.reset()
        return reward, done
    
    def reset(self):
        self.obs = self.env.reset()