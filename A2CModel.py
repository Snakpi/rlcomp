# -*- coding: utf-8 -*-
from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd


# Actor Critic architecture setup
class ActorCritic(nn.Module): 
    
    def __init__(
            self,
            num_inputs, #The number of inputs
            action_space, #The number of actions
            gamma = 0.99, #The discount factor
            epsilon = 1, #Epsilon - the exploration factor
            epsilon_min = 0.01, #The minimum epsilon 
            epsilon_decay = 0.999,#The decay epislon for each update_epsilon time
            learning_rate = 0.001, #The learning rate for the DQN network 
            hidden_sizes = [100, 100, 50],
            cuda = True
    ):
      super(ActorCritic, self).__init__()
    
      self.num_inputs = num_inputs
      self.action_space = action_space
      self.gamma = gamma
      self.epsilon = epsilon
      self.epsilon_min = epsilon_min
      self.epsilon_decay = epsilon_decay
      self.learning_rate = learning_rate
      self.hidden_sizes = hidden_sizes
            
      #Creating networks
      self.critic = self.create_NN(num_inputs, hidden_sizes, 1)
      self.actor = self.create_NN(num_inputs, hidden_sizes, action_space)
      self.cuda = cuda

      if self.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
      else:
        device = torch.device('cpu')
      
    
    def create_NN(self, 
      num_inputs=self.num_inputs, 
      hidden_sizes=self.hidden_sizes, 
      num_outputs=self.action_space):
      '''
      Creating NN given inputs num, hidden_sizes, and output num
      (default are those for actor i.e outputs == action_space)
      '''
      layers = []

      layers.append(nn.Linear(num_inputs, hidden_sizes[0]))
      layers.append(nn.ReLU())

      for i in range(len(hidden_sizes)):
        layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        layers.append(nn.ReLU())

      layers.append(nn.Linear(hidden_sizes[-1], num_outputs))
      layers.append(nn.Softmax())

      net = nn.Sequential(*layers)
      return net

    
    def forward(self, state):
      ''' Forward pass and returns the value & policy dist '''
      state = Variable(torch.from_numpy(state).float().unsqueeze())
      value = self.critic(state)
      policy_dist = self.actor(state)
      return value, policy_dist
    

    def update_epsilon(self):
      self.epsilon =  self.epsilon*self.epsilon_decay
      self.epsilon =  max(self.epsilon_min, self.epsilon)
      
      
    def save_actor(self, path, actor_name):
        # save model
        torch.save(self.actor.state_dict(), path + actor_name + '.pth')
        # serialize weights to HDF5
        print("Saved actor to disk")