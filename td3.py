from networks import Actor, Critic
import torch 
import numpy as np 
import torch.nn.functional as F
import copy 

class TD3(object):
    def __init__(self, state_dim, action_dim, hidden_dim, limit, is_adv, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2, expl_noise=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #init actors
        self.actor = Actor(state_dim, hidden_dim, action_dim, limit).to(self.device)
        self.target_actor = copy.deepcopy(self.actor) #frozen actor
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        
        #init critics
        self.critic = Critic(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_critic = copy.deepcopy(self.critic) #frozen critic
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.limit = limit
        self.discount = discount
        self.is_adv = is_adv
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.expl_noise = expl_noise

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1
        s, a, r, ns, d = replay_buffer.sample(batch_size, self.is_adv)
        state = torch.FloatTensor(s).to(self.device)
        action = torch.FloatTensor(a).to(self.device)
        next_state = torch.FloatTensor(ns).to(self.device)
        reward = torch.FloatTensor(r).to(self.device)
        not_done = torch.FloatTensor(1. - d).to(self.device)

        with torch.no_grad():

            noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.target_actor(next_state) + noise).clamp(-self.limit, self.limit)

            target_q1, target_q2 = self.target_critic(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.discount * target_q

        current_q1, current_q2 = self.critic(state, action)
        
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic.pt")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer.pt")
        torch.save(self.actor.state_dict(), filename + "_actor.pt")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer.pt")
    
    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic.pt"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer.pt"))
        self.target_critic = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor.pt"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer.pt"))
        self.target_actor = copy.deepcopy(self.actor)
		