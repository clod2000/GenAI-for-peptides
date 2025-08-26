#########################################################################################
#                                                                                       #
#  Variational Graph AutoEncoder (VGAE) for dihedral angles                             #
#                                                                                       #
#  Author: Claudio Colturi                                                              #
#                                                                                       #
#########################################################################################



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
from torch_geometric.nn import SAGEConv, GATConv



############ encoders ############

class GCN_encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers=2, attention=False,heads=1, batch_norm = False):
        super(GCN_encoder, self).__init__()

        self.norm = torch.nn.BatchNorm1d(hidden_channels) if batch_norm else torch.nn.Identity()
        
        for i in range(num_layers - 1):
            if i == 0:
                setattr(self, f'conv{i+1}', GCNConv(in_channels, hidden_channels))
            else:
                setattr(self, f'conv{i+1}', GCNConv(hidden_channels, hidden_channels))
                if attention:
                    setattr(self, f'attn{i+1}', GATConv(hidden_channels, hidden_channels,heads= heads))

                    

        
        self.linear_mu = torch.nn.Linear(hidden_channels, out_channels)
        self.linear_logstd = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # Apply convolution and optional attention layers
        for i in range(self.num_layers):
            #print(f"x_shape: {x.shape}")
            x = getattr(self, f'conv{i+1}')(x, edge_index)
            x = self.norm(x)
            x = F.leaky_relu(x)

            if self.attention:
                x = getattr(self, f'attn{i+1}')(x, edge_index)

        x = global_mean_pool(x, batch)  # Pool node features to get graph-level embedding

        # calculate mu and logstd from the single graph-level embedding
        mu = self.linear_mu(x)
        logstd = self.linear_logstd(x)
       
        return mu, logstd

class SAGE_encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers=2, attention=False, heads=1, batch_norm=False):
        super(SAGE_encoder, self).__init__()
        
        self.num_layers = num_layers
        self.attention = attention
        self.norm = torch.nn.BatchNorm1d(hidden_channels) if batch_norm else torch.nn.Identity()

        self.skip_connection = True  # Enable skip connections 
        
        # Use ModuleLists to correctly register layers
        self.convs = torch.nn.ModuleList()
        if attention:
            self.attns = torch.nn.ModuleList()

        for i in range(num_layers):
            layer_in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(layer_in_channels, hidden_channels))

            if attention:
                # FIX: Intermediate layers average, the FINAL layer concatenates.
                is_final_layer = (i == num_layers - 1)
                # The GAT layer always takes the output of the SAGE layer as input
                self.attns.append(GATConv(hidden_channels, hidden_channels, heads=heads, concat=is_final_layer))

        # FIX: The final dimension depends on whether the last GAT layer concatenated.
        final_embedding_dim = hidden_channels * heads if attention else hidden_channels
        self.linear_mu = torch.nn.Linear(final_embedding_dim, out_channels)
        self.linear_logstd = torch.nn.Linear(final_embedding_dim, out_channels)

    def forward(self, x, edge_index, batch):
        # FIX: Implement a sequential forward pass
        for i in range(self.num_layers):
            x_input = x  # Store input for potential skip connection

            # Pass the output of the previous layer as input to the current one
            x = self.convs[i](x, edge_index)
            
            # Apply attention after the convolution
            if self.attention:
                x = self.attns[i](x, edge_index)

            
            x_out = self.norm(x)
        
            
            if self.skip_connection and x_input.shape == x.shape:
                x = x_out + x_input

        
            else:
                # This can happen if the first layer changes the channel size.
                # In that case, you might skip the residual on the first layer or project x_input.
                x = x_out
                
            if i < self.num_layers - 1 or not self.attention:
                x = self.norm(x)
                x = F.leaky_relu(x)

        # Global pooling happens after all layers are done
        x = global_mean_pool(x, batch)

        # Calculate mu and logstd from the final graph-level embedding
        mu = self.linear_mu(x)
        logstd = self.linear_logstd(x)

        return mu, logstd

# class SAGE_encoder(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, hidden_channels, num_layers=2, attention=False, heads=1, batch_norm=False):
#         super(SAGE_encoder, self).__init__()
        
#         # --- Store parameters that are needed in forward pass ---
#         self.num_layers = num_layers
#         self.attention = attention
#         self.norm = torch.nn.BatchNorm1d(hidden_channels) if batch_norm else torch.nn.Identity()
        
#         # Initialize layers
#         for i in range(num_layers):
#             print(f"Initializing layer {i+1}")
#             is_final_layer = (i == num_layers - 1)
#             layer_in_channels = in_channels if i == 0 else hidden_channels
#             # Correctly handle input channels for the first layer
#             setattr(self, f'conv{i+1}', SAGEConv(layer_in_channels, hidden_channels))
#             if attention:
#                 # The input to GATConv should be the output of SAGEConv
#                 setattr(self, f'attn{i+1}', GATConv(layer_in_channels, hidden_channels, heads=heads, concat= not is_final_layer))

#         # The input to these linear layers is now the pooled graph embedding
#         # If using attention with multiple heads, the dimension will be larger
#         final_embedding_dim = hidden_channels * heads if attention else hidden_channels
#         self.linear_mu = torch.nn.Linear(final_embedding_dim, out_channels)
#         self.linear_logstd = torch.nn.Linear(final_embedding_dim, out_channels)

#     def forward(self, x, edge_index, batch):
#         # Apply convolution and optional attention layers
#         for i in range(self.num_layers):
#             # #print(f"x_shape: {x.shape}")
#             # x = getattr(self, f'conv{i+1}')(x, edge_index)
#             # x = self.norm(x)
#             # x = F.leaky_relu(x)

#             # if self.attention:
#             #     x = getattr(self, f'attn{i+1}')(x, edge_index)
#             x_input = x
#             # Calculate SAGE and GAT in parallel
#             x_from_sage = getattr(self, f'conv{i+1}')(x_input, edge_index)
#             if self.attention:
#                 x_from_gat = getattr(self, f'attn{i+1}')(x_input, edge_index)
            
#         if self.attention: 
#             x = x_from_sage + x_from_gat
#         else:
#             x = x_from_sage


#             # Combine their outputs by adding them
#         x = F.leaky_relu(self.norm(x))
        

#         x = global_mean_pool(x, batch)  # Pool node features to get graph-level embedding

#         # calculate mu and logstd from the single graph-level embedding
#         mu = self.linear_mu(x)
#         logstd = self.linear_logstd(x)

#         return mu, logstd


class MLP_Decoder(torch.nn.Module):
    def __init__(self, latent_dim, out_nodes, out_features, hidden_channels, num_layers=3):
        super().__init__()
    
        self.out_nodes = out_nodes
        self.out_features = out_features
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.skip_connections = True

        layers = []
        layers.append(torch.nn.Linear(latent_dim, hidden_channels))
        layers.append(torch.nn.SiLU())  # added SiLU activation after the first layer and changed ReLU to SiLU in the other layers
        for i in range(num_layers - 1):
            layers.append(torch.nn.Linear(hidden_channels, hidden_channels))
            layers.append(torch.nn.SiLU())
        layers.append(torch.nn.Linear(hidden_channels, out_nodes * out_features*2)) # added a 2
        self.fc = torch.nn.Sequential(*layers)

    def forward(self, z):

        # x_recon = fc(z)
        # x_recon = torch.tanh(x_recon) #expect the output to be in the range [-1, 1]
        #return x_recon.view(-1, self.out_nodes, self.out_features)  # shape: [batch, nodes, features]

    
        
        logits = self.fc(z)
        reshaped_logits = logits.view(-1, self.out_nodes, self.out_features, 2)
        
        x_coords = reshaped_logits[..., 0]
        y_coords = reshaped_logits[..., 1]
        
        x_recon_radians = torch.atan2(y_coords, x_coords)        
       
        return x_recon_radians #.view(-1, self.out_nodes, self.out_features)  # shape: [batch, nodes, features]



class DVGAE(torch.nn.Module):
    def __init__(self, encoder, decoder, device = torch.device('cpu')):
        super(DVGAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.latent_dim = encoder.linear_mu.out_features
        self.device = device


    def encode(self, x, edge_index, batch):
        mu, logstd = self.encoder(x, edge_index, batch)
        return mu, logstd

    def decode(self, z):
        return self.decoder(z)
    
    def reparametrize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, edge_index, batch):
        mu, logstd = self.encode(x, edge_index, batch)
        z = self.reparametrize(mu, logstd)
        x_recon = self.decode(z)
        return x_recon, mu, logstd

    
