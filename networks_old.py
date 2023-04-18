import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class Actor(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, limit):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(d_in, d_hidden)
        self.layer2 = nn.Linear(d_hidden, d_hidden)
        self.mu = nn.Linear(d_hidden, d_out)
        self.sigma = nn.Linear(d_hidden, d_out)
        self.limit = limit

        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, weight_decay=1e-4)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

        nn.init.xavier_uniform_(self.mu.weight)
        nn.init.xavier_uniform_(self.sigma.weight)

    def forward(self, state):
        a = F.relu(self.layer1(state))
        a = F.relu(self.layer2(a))
        mu =  torch.sigmoid(self.mu(a))
        sigma = F.softplus(self.sigma(a))
        
        return mu, sigma #outputs a categorical distribution according to probabilites 
    
    def get_dist(self, state):
        mu, sigma = self.forward(state)
        dist = torch.distributions.normal.Normal(mu, sigma)
        return dist
        
    
class Critic(nn.Module):
    def __init__(self, d_in, d_hidden):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(d_in, d_hidden)
        self.l2 = nn.Linear(d_hidden,d_hidden)
        self.l3 = nn.Linear(d_hidden, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state):
        value = torch.tanh(self.l1(state))
        value = torch.tanh(self.l2(value))
        value = self.l3(value)
        
        return value

        