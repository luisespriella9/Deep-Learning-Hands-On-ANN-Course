# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# create the architecture of the neural network

class NeuralNetwork(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.full_connection_1 = nn.Linear(input_size, 30)
        self.full_connection_2 = nn.Linear(30, 20)
        self.full_connection_3 = nn.Linear(20, nb_action)
        
    def forward(self, input_state):
        hidden_neurons1 = functional.relu(self.full_connection_1(input_state)) #Relu activation function
        hidden_neurons2 = functional.relu(self.full_connection_2(hidden_neurons1)) #Relu activation function
        q_values = self.full_connection_3(hidden_neurons2)
        return q_values
    
#implementing Experience Replay

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self, event):
        #event is a tuple of 4 objects: last state, new state, last action and last reward
        self.memory.append(event)
        if (len(self.memory) > self.capacity):
            del self.memory[0]
            
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
    
#Implementing Deep Q Learning

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma, learning_rate = 0.001):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.reward_window = []
        self.network_model = NeuralNetwork(input_size, nb_action)
        self.replay_memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.network_model.parameters(), lr = learning_rate)
        self.last_state = torch.Tensor(input_size).unsqueeze(0) #fake dimension to index 0
        self.last_action = 0
        self.last_reward = 0
        
    def select_action(self, input_state):
        action_probabilities = functional.softmax(self.network_model(Variable(input_state, volatile=True))*75) 
        action = action_probabilities.multinomial()
        return action.data[0, 0]
        
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        #calculate state
        outputs = self.network_model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        #next state
        next_outputs = self.network_model(batch_next_state).detach().max(1)[0]
        #target
        target = self.gamma*next_outputs + batch_reward
        #calculate temporal difference
        temporal_diff_loss = functional.smooth_l1_loss(outputs, target)
        #reinitialize optimizer
        self.optimizer.zero_grad()
        #backpropagate error into neural network
        temporal_diff_loss.backward(retain_variables=True)
        #update weights
        self.optimizer.step()
        
    def update(self, reward, new_signal):
        #when reaching a new state
        new_state = torch.Tensor(new_signal).float().unsqueeze(0) #and add fake dimension to torch tensor
        #update memory after reaching new state
        self.replay_memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        #play an action
        action = self.select_action(new_state)
        if (len(self.replay_memory.memory) > 100):
            #learn from 100 transitions of memory
            batch_state, batch_next_state, batch_action, batch_reward = self.replay_memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        #update last action
        self.last_action = action
        #update last state
        self.last_state = new_state
        #update reward
        self.last_reward = reward
        self.reward_window.append(reward)
        if (len(self.reward_window) > 1000):
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/len(self.reward_window)+1
    
    def save(self):
        torch.save({'state_dict': self.network_model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                    }, 'last_brain.pth')
    
    def load(self):
        #if file exists, then load
        if (os.path.isfile('last_brain.pth')):
            print("loading checkpoint ...")
            checkpoint = torch.load('last_brain.pth')
            #update existing model and optimizer with parameters and weights
            self.network_model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Checkpoint Loaded")
        else:
            print("No checkpoint found")