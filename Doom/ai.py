# AI for Doom



# Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Importing the packages for OpenAI and Doom
import gym
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

# Importing the other Python files
import experience_replay, image_preprocessing



# Part 1 - Building the AI

# making the brain

class CNN(nn.Module):
    
    def __init__(self, n_actions):
        super(CNN, self).__init__()
        self.n_actions = n_actions
        #1 channel for black and white images as we will only be detecting size
        #out channels -> number of features we want to detect
        #kernel size 5x5 dimension
        self.input_convolution_layer = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5) #2d images
        self.hidden_layer = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.output_convolution_layer = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2)
        self.input_to_hidden_full_connection = nn.Linear(in_features =  self.count_neurons((1, 80, 80)), out_features = 40)
        self.hidden_to_output_full_connection = nn.Linear(in_features =  40, out_features = n_actions)
        
    def count_neurons(self, image_dimensions):
        #create fake image with right dimensions
        fake_image = Variable(torch.rand(1, *image_dimensions)) #put random pixels
        #flatten image to get neurons; convolution->max pooling-> activate neurons -> get flattening layer
        fake_image = F.relu(F.max_pool2d(self.input_convolution_layer(fake_image), kernel_size=3, stride=2))
        fake_image = F.relu(F.max_pool2d(self.hidden_layer(fake_image), kernel_size=3, stride=2))
        fake_image = F.relu(F.max_pool2d(self.output_convolution_layer(fake_image), kernel_size=3, stride=2))
        return fake_image.data.view(1, -1).size(1)
    
    def forward(self, input_image):
        input_image = F.relu(F.max_pool2d(self.input_convolution_layer(input_image), kernel_size=3, stride=2))
        input_image = F.relu(F.max_pool2d(self.hidden_layer(input_image), kernel_size=3, stride=2))
        input_image = F.relu(F.max_pool2d(self.output_convolution_layer(input_image), kernel_size=3, stride=2))
        input_image = input_image.view(input_image.size(0), -1) #flatten layer of multiple channels
        input_image = F.relu(self.input_to_hidden_full_connection(input_image))
        input_image = self.hidden_to_output_full_connection(input_image)
        return input_image

# make the body

class SoftMaxBody(nn.Module):
    
    def __init__(self, confidence):
        super(SoftMaxBody, self).__init__()
        self.confidence = confidence
        
    def forward(self, outputs):
        probabilities = F.softmax(outputs*self.confidence)
        actions = probabilities.multinomial()
        return actions

# making the AI

class AI(nn.Module):
    
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body
        
    def __call__(self, input_images):
        nn_input = Variable(torch.from_numpy(np.array(input_images, dtype = np.float32)))
        output = self.brain.forward(nn_input)
        actions = self.body.forward(output)
        return actions.data.numpy()
        
        
# Part 2 - Implementing Deep Convolutional Q-Learning

# getting the doom environment

doom_env = image_preprocessing.PreprocessImage(( gym.wrappers.SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0")))), 80, 80, grayscale=True)
doom_env = gym.wrappers.Monitor(doom_env, "videos", force=True)
number_actions = doom_env.action_space.n

# building an AI

cnn = CNN(number_actions)
softmax_body = SoftMaxBody(1.0)
ai = AI(cnn, softmax_body)

# setting up experience replay

n_steps = experience_replay.NStepProgress(env = doom_env, ai = ai, n_step = 10)
memory = experience_replay.ReplayMemory(n_steps = n_steps, capacity = 10000)

# implementing eligibility trace

def in_step_q_learning(batch_input_images):
    discount_function = 0.99
    inputs = []
    targets = []
    for input_images in batch_input_images:
        input_states = Variable(torch.from_numpy(np.array([input_images[0].state, input_images[-1].state], dtype=np.float32)))
        output = cnn.forward(input_states)
        #compute cumulative reward
        cumul_reward = 0.0 if input_images[-1].done else output[1].data.max()
        for step in reversed(input_images[:-1]):
            cumul_reward = cumul_reward*discount_function + step.reward
        state = input_images[0].state
        target = output[0].data #get q value from first step of transition
        target[input_states[0].action] = cumul_reward
        inputs.append(state) #add first step of the series
        targets.append(target)
    return torch.from_numpy(np.array(inputs, dtype = np.float32)), torch.stack(targets)

# making the moving average on 100 steps (stochastic gradient descent)

class MA:
    
    def __init__(self, size):
        self.size = size
        self.list_of_rewards = []
        
    def add_cumul_reward(self, rewards):
        if (isinstance(rewards, list)):
            self.list_of_rewards += rewards       
        else:
            self.list_of_rewards.append(rewards)
        while (self.list_of_rewards > self.size):
            del self.list_of_rewards[0]
            
    def average(self):
        return np.mean(self.list_of_rewards)
    
ma = MA(100)

#training the AI

loss = nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr = 0.001)
num_epochs = 100

for epoch in range(1, num_epochs+1):
    memory.run_steps(200)
    for batch in memory.sample_batch(128): #usually 32, however with multiple steps we want to take more in this case
        inputs, targets = in_step_q_learning(batch)
        inputs, targets = Variable(inputs), Variable(targets)
        predictions = cnn(inputs)
        loss_error = loss(predictions, targets)
        #backpropagate
        optimizer.zero_grad() #initialize optimizer
        loss_error.backward()
        optimizer.step()
    reward_steps = n_steps.rewards_steps()
    ma.add_cumul_reward(reward_steps)
    avg_reward = ma.average()
    print("Epoch: %s, Average Reward %s" % (str(epoch), str(avg_reward)))
    if (avg_reward >= 1500):
        print("Congratulations! your AI wins")
        break
    
#closing the environment
doom_env.close()