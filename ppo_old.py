from networks_old import Actor, Critic
from ppo_memory import PPOMemory
import torch 
import numpy as np 
import copy
import math

class PPO(object):
    def __init__(self, state_dim, action_dim, hidden_dim, limit, is_adv, gamma=0.99, gae_lambda=0.95, policy_clip=0.2, batch_size=512, N=128, n_epochs=10, expl_noise=0.2, entropy_coeff=0, entropy_decay=0.9998, l2_reg=1e-3):
        self.limit = limit
        self.gamma = gamma
        self.is_adv = is_adv
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.batch_size = batch_size
        self.N = N #number of steps before update (horizon)
        self.n_epochs =n_epochs
        self.expl_noise = expl_noise
        self.entropy_coeff = entropy_coeff
        self.entropy_decay = entropy_decay
        self.l2_reg = l2_reg

        #init agent
        self.actor_network = Actor(state_dim, hidden_dim, action_dim, self.limit)
        self.critic_network = Critic(state_dim, hidden_dim)
        self.memory = []
        self.total_it = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    
    
    def store_memory(self, state, action, reward, next_state, log_prob, done):
        self.memory.append([state, action, reward, next_state, log_prob, done])
    
    def select_action(self, state): #returns action arr, log prob arr, and value arr
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float).to(self.actor_network.device)
                dist = self.actor_network.get_dist(state)
                actions = dist.sample()
                actions = torch.clamp(actions, 0, 1)
                log_prob_actions = dist.log_prob(actions).cpu().numpy().flatten()
                return actions.cpu().numpy().flatten(), log_prob_actions

    
    def train(self):
            self.entropy_coeff*=self.entropy_decay
            s, a, r, ns, logprob, done = self.make_batch()

            with torch.no_grad():
                vs = self.critic_network(s)
                vns = self.critic_network(ns)


                 #compute deltas
                deltas = r + self.gamma * vns - vs

                deltas = deltas.cpu().flatten().numpy()

                adv = [0]

                for dlt, mask in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
                    advantage = dlt + self.gamma * self.gae_lambda * adv[-1] * (1 - mask) #compute advantage for run
                    #adv = np.clip(adv, a_min=-10, a_max=10).tolist()
                    adv.append(advantage)
                adv.reverse()
                adv = copy.deepcopy(adv[0:-1])
                adv = torch.tensor(adv).unsqueeze(1).float().to(self.device)
                td_target = adv + vs
                if len(adv) > 1:
                    adv = (adv - adv.mean()) / ((adv.std()+1e-4))  #sometimes helps
            
            #perform mini-batch PPO update by slicing long trajectories
            a_optim_iter_num = int(math.ceil(s.shape[0] / self.batch_size))
            c_optim_iter_num = int(math.ceil(s.shape[0] / self.batch_size)) #TODO, change batch size for each optimizer

            for i in range(self.n_epochs):  
            
                #shuffle trajectory
                permutation = np.arange(s.shape[0])
                np.random.shuffle(permutation)
                permutation = torch.LongTensor(permutation).to(self.device)

                s, a, td_target, adv, logprob = \
                    s[permutation].clone(), a[permutation].clone(), td_target[permutation].clone(), adv[permutation].clone(), \
                    logprob[permutation].clone()
                
                #training the actor
                for i in range(a_optim_iter_num):
                    index = slice(i*self.batch_size, min((i+1) * self.batch_size, s.shape[0])) 
                    dist = self.actor_network.get_dist(s[index])
                    dist_entropy = dist.entropy().sum(1, keepdim=True)
                    logprob_new = dist.log_prob(a[index])
                    prob_ratio = torch.exp(logprob_new.sum(1,keepdim=True) - logprob[index].sum(1,keepdim=True))

                    surrogate_loss1 = prob_ratio*adv[index]
                    surrogate_loss2 = torch.clamp(prob_ratio, 1- self.policy_clip, 1 + self.policy_clip) * adv[index]
                    actor_loss = -torch.min(surrogate_loss1,surrogate_loss2) - self.entropy_coeff * dist_entropy

                    self.actor_network.optimizer.zero_grad()
                    actor_loss.mean().backward()
                    #print("ACTOR LOSS: " + str(actor_loss.mean()))
                    torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), 40) #clip gradients to prevent explosions/nans
                    self.actor_network.optimizer.step()
                
                for i in range(c_optim_iter_num):
                    index = slice(i*self.batch_size, min((i+1) * self.batch_size, s.shape[0])) 
                    critic_loss = (self.critic_network(s[index]) - td_target[index]).pow(2).mean()
                    for name, param in self.critic_network.named_parameters():
                        if 'weight' in name:
                            critic_loss += param.pow(2).sum() * self.l2_reg #calculate sum of squared differences error with l2_reg
                    
                    self.critic_network.optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_network.optimizer.step()

                

    def make_batch(self):
        s_l, a_l, r_l, ns_l, logprob_l, done_l = [], [], [], [], [], []
        for transition in self.memory:
            s, a, r, ns, logprob, done = transition

            s_l.append(s)
            a_l.append(a)
            logprob_l.append(logprob)
            r_l.append([r])
            ns_l.append(ns)
            done_l.append([done])

        if len(self.memory) >= self.N:
            self.memory = [] #only clear the memory if we've reached maximum batch size

        with torch.no_grad():
            s, a, r, ns, logprob, done = \
            torch.tensor(np.array(s_l), dtype=torch.float).to(self.device), \
            torch.tensor(np.array(a_l), dtype=torch.float).to(self.device), \
            torch.tensor(np.array(a_l), dtype=torch.float).to(self.device), \
            torch.tensor(np.array(ns_l), dtype=torch.float).to(self.device), \
            torch.tensor(np.array(logprob_l), dtype=torch.float).to(self.device), \
            torch.tensor(np.array(done_l), dtype=torch.float).to(self.device)

        
        return s, a, r, ns, logprob, done
            

    def save(self, filename):
        print('... saving models ...')
        torch.save(self.critic_network.state_dict(), filename + "_critic.pt")
        torch.save(self.critic_network.optimizer.state_dict(), filename + "_critic_optimizer.pt")
        torch.save(self.actor_network.state_dict(), filename + "_actor.pt")
        torch.save(self.actor_network.optimizer.state_dict(), filename + "_actor_optimizer.pt")
    
    def load(self, filename):
        self.critic_network.load_state_dict(torch.load(filename + "_critic.pt"))
        self.critic_network.optimizer.load_state_dict(torch.load(filename + "_critic_optimizer.pt"))
        self.actor_network.load_state_dict(torch.load(filename + "_actor.pt"))
        self.actor_network.optimizer.load_state_dict(torch.load(filename + "_actor_optimizer.pt"))
		