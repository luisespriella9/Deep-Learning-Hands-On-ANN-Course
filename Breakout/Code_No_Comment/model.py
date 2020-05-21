# AI for Breakout

# Importing the librairies
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Initializing and setting the variance of a tensor of weights
def normalized_columns_initializer(weights, std = 1.0):
    output_weights = torch.randn(weights.size()) #randomize torch variable
    output_weights *= std / torch.sqrt(output_weights.pow(2).sum(1).expand_as(output_weights)) #normalize
    return output_weights

# Initializing the weights of the neural network in an optimal way for the learning
def weights_init(model):
    classname = model.class__.__name__
    # if convolution
    if (classname.find("Conv") != -1):
        #get weight shape
        weight_shape = list(model.weight.data.size())
        #initialize weights for convolution
        fan_in = np.prod(weight_shape[1:4]) #dim1 * dim2 * dim3
        fan_out = weight_shape[0]*np.prod(weight_shape[2:4]) #dim0 * dim2 * dim3
        #initialize weights for convolution
        w_bound = np.sqrt(6. / fan_in + fan_out)
        model.weight.data.uniform_(-w_bound, w_bound)
        model.bias.data.fill_(0) #fill bias with zero
    # if Linear
    elif (classname.find("Linear") != -1):
        #get weight shape
        weight_shape = list(model.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / fan_in + fan_out)
        model.weight.data.uniform_(-w_bound, w_bound)
        model.bias.data.fill_(0) #fill bias with zero

# Making the A3C brain

class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        # convolutions
        self.conv1 = nn.Conv2D(in_channels=num_inputs, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size3, stride=2, padding=1)
        self.conv3 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size3, stride=2, padding=1)
        self.conv4 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size3, stride=2, padding=1)
        
        #LSTM
        #input shape = 32*3*3, output shape = 256
        self.lstm = nn.LSTMCell(input_size=32*3*3, hidden_size=256)
        num_actions = action_space.n
        # full connection for critic
        self.critic_full_connection = nn.Linear(in_features=256, out_features=1)
        # full connection for actor
        self.actor_full_connection = nn.Linear(in_features=256, out_features=num_actions)
        
        # initialize weights of neural networks
        self.apply(weights_init)
        #small std for actor and large for critic to manage explore vs explot
        self.actor_full_connection.weight.data = normalize_columns_initializer(self.actor_full_connection.weight.data, std=0.01)
        self.critic_full_connection.weight.data = normalize_columns_initializer(self.critic_full_connection.weight.data, std=1)
        
        #initialize bias of LSTM
        self.lstm.bias_ih.data.fill(0)
        self.lstm.bias_hh.data.fill(0)
        self.train() #put model in train mode
        
    def forward(self, inputs):
        # split inputs for images for convolution and states for LSTM
        input_images, (hidden_states, cell_states) = inputs
        # forward through convolutional layers
        first_conv_layer = F.elu(self.conv1(input_images))
        second_conv_layer= F.elu(self.conv2(first_conv_layer))
        third_conv_layer = F.elu(self.conv3(second_conv_layer))
        fourth_conv_layer = F.elu(self.conv4(third_conv_layer))
        
        #flatten
        flatter_layer = fourth_conv_layer.view(-1, 32*3*3)
        
        #forward through LSTM
        (output_hidden_nodes, cell_nodes) = self.lstm(flatten_layer, (hidden_states, cell_states))
        
        return self.critic_full_connection(output_hidden_nodes), self.actor_full_connection(output_hidden_nodes), (output_hidden_nodes, cell_nodes)