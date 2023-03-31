import torch
from torch import nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, limit):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(d_in, d_hidden)
        self.layer2 = nn.Linear(d_hidden, d_hidden)
        self.layer3 = nn.Linear(d_hidden, d_out)
        self.limit = limit #max value of continuous action

    def forward(self, state):
        a = F.relu(self.layer1(state))
        a = F.relu(self.layer2(a))
        return torch.tanh(self.layer3(a)) * self.limit 
    
class Critic(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super(Critic, self).__init__()

        #Q1 
        self.layer1 = nn.Linear(d_in + d_out, d_hidden)
        self.layer2 = nn.Linear(d_hidden, d_hidden)
        self.layer3 = nn.Linear(d_hidden, 1)

        #Q2
        self.layer4 = nn.Linear(d_in + d_out, d_hidden)
        self.layer5 = nn.Linear(d_hidden, d_hidden)
        self.layer6 = nn.Linear(d_hidden, 1)

    def forward(self, state, action):
        sa = torch.cat([state,action], 1)

        q1 = F.relu(self.layer1(sa))
        q1 = F.relu(self.layer2(q1))
        q1 = self.layer3(q1)
        
        q2 = F.relu(self.layer4(sa))
        q2 = F.relu(self.layer5(q2))
        q2 = self.layer6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state,action], 1)

        q1 = F.relu(self.layer1(sa))
        q1 = F.relu(self.layer2(q1))
        q1 = self.layer3(q1)
        return q1
        