import torch_geometric as pyg
import torch._numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
from torch_geometric.nn.pool import global_mean_pool

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges

import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx
import networkx as nx

from torch_geometric.nn import VGAE
from torch_geometric.nn import SAGEConv


############ encoders ############

class GCN_encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers=2):
        super(GCN_encoder, self).__init__()
        
        for i in range(num_layers - 1):
            if i == 0:
                setattr(self, f'conv{i+1}', GCNConv(in_channels, hidden_channels))
            else:
                setattr(self, f'conv{i+1}', GCNConv(hidden_channels, hidden_channels))
        
        self.linear_mu = torch.nn.Linear(hidden_channels, out_channels)
        self.linear_logstd = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index,batch):
        for i in range(self.num_layers - 1):
            x = getattr(self, f'conv{i+1}')(x, edge_index)
            x = torch.leaky_relu(x)
        mu = self.linear_mu(x)
        logstd = self.linear_logstd(x)

        x = global_mean_pool(x, batch) # Pool node features to get graph-level embedding
    
        return mu, logstd
    
class SAGE_encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers=2):
        super(SAGE_encoder, self).__init__()
        self.num_layers = num_layers
        for i in range(num_layers - 1):
            if i == 0:
                setattr(self, f'conv{i+1}', SAGEConv(in_channels, hidden_channels))
            else:
                setattr(self, f'conv{i+1}', SAGEConv(hidden_channels, hidden_channels))

        self.linear_mu = torch.nn.Linear(hidden_channels, out_channels)
        self.linear_logstd = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for i in range(self.num_layers - 1):
            x = getattr(self, f'conv{i+1}')(x, edge_index)
            x = torch.relu(x)
        x = getattr(self, f'conv{self.num_layers}')(x, edge_index)
        x = torch.relu(x)

        #apply a global pooling
        x = global_mean_pool(x, batch) # Pool node features to get graph-level embedding
       
        mu = self.linear_mu(x)
        logstd = self.linear_logstd(x)
       
        return mu, logstd


############# Decoders ############

#decoder to reconstruct the angles from the latent space

# class MLP_Decoder(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(MLP_Decoder, self).__init__()
#         self.fc1 = torch.nn.Linear(in_channels, hidden_channels[0])
#         self.fc2 = torch.nn.Linear(hidden_channels[0], hidden_channels[1])
#         self.fc3 = torch.nn.Linear(hidden_channels[1], out_channels)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
    
#

class MLP_Decoder(torch.nn.Module):
    def __init__(self, latent_dim, out_nodes, out_features, hidden_channels, num_layers=3):
        super().__init__()
    
        self.out_nodes = out_nodes
        self.out_features = out_features
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        layers = []
        layers.append(torch.nn.Linear(latent_dim, hidden_channels))
        for i in range(num_layers - 1):
            layers.append(torch.nn.Linear(hidden_channels, hidden_channels))
            layers.append(torch.nn.LeakyReLU())
        layers.append(torch.nn.Linear(hidden_channels, out_nodes * out_features))
        self.fc = torch.nn.Sequential(*layers)

    def forward(self, z):
        x_recon = self.fc(z)
        x_recon = torch.tanh(x_recon) #expect the output to be in the range [-1, 1]
        return x_recon.view(-1, self.out_nodes, self.out_features)  # shape: [batch, nodes, features]



# useful functions

def angle_loss(pred, target):  # use sin and cos to compute the angle loss to avoid discontinuity 
    if pred.shape[2] == 1:
        
        return F.mse_loss(torch.sin(pred), torch.sin(target)) + F.mse_loss(torch.cos(pred), torch.cos(target))
    else: # if already in sin and cos
        return F.mse_loss(pred, target)

def compute_loss(model, x_recon, x, beta=1.0, learn_lambda=False):
    

    angles_loss = angle_loss(x_recon, x)
    kl_loss = model.kl_loss()  # KL divergence loss for regularization
    if learn_lambda:
        #beta = lambda_reg
        return beta*angles_loss +(1-beta)*kl_loss
    else:
        return angles_loss + beta*kl_loss


def inverse_transform(x): # to go back from sin and cos to angle
    x = torch.clamp(x, -1, 1)  # clamp values to avoid NaN in atan2
    x_sin = x[:, :, 0]
    x_cos = x[:, :, 1]
    x = torch.atan2(x_sin, x_cos)
    x = torch.rad2deg(x)  # convert radians to degrees
    
    return x


def beta_annealer(epochs,beta_start = 0., beta_end = 1., annealing_epochs = 100, wait_epochs = 10):
    
    if epochs < wait_epochs:
        return beta_start
    
    return beta_start + (beta_end - beta_start) * min(1,((epochs-wait_epochs) / annealing_epochs))
