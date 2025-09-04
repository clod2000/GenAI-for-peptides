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


class encoder(torch.nn.Module):
    """
    Graph Neural Network (GNN) encoder for Variational Graph Autoencoder (VGAE).
    Supports GCN, GraphSAGE, and GAT layers with optional skip connections and batch normalization.
    Parameters:
    - in_channels (int): Number of input features per node.
    - out_channels (int): Dimensionality of the latent space (output features).
    - hidden_channels (int): Number of hidden units in each GNN layer.
    - enc_type (str): Type of GNN layer to use ('GCN', 'SAGE', 'GAT').
    - num_layers (int): Number of GNN layers.
    - attention (bool): Whether to use attention mechanism (only for GAT).
    - heads (int): Number of attention heads (only for GAT).
    - batch_norm (bool): Whether to apply batch normalization after each layer.
    - skip_connection (bool): Whether to use skip connections between layers. ( For now always True)
    Returns:
    - mu (Tensor): Mean of the latent space distribution.
    - logstd (Tensor): Log standard deviation of the latent space distribution. 

    Note: The final embedding dimension depends on whether the last GAT layer concatenates outputs.
    """
    def __init__(self, in_channels, out_channels, hidden_channels, enc_type = 'SAGE', num_layers=2, attention=False, heads=1, batch_norm=False):
        super(encoder, self).__init__()
        
        self.num_layers = num_layers
        self.norm = torch.nn.BatchNorm1d(hidden_channels) if batch_norm else torch.nn.Identity()
        self.skip_connection = True  # Enable skip connections 
        self.convs = torch.nn.ModuleList()
      
        for i in range(num_layers):
            layer_in_channels = in_channels if i == 0 else hidden_channels
            if enc_type == 'GCN':
                self.convs.append(GCNConv(layer_in_channels, hidden_channels))
            elif enc_type == 'SAGE':
                self.convs.append(SAGEConv(layer_in_channels, hidden_channels))
            elif enc_type == 'GAT':
                self.convs.append(GATConv(layer_in_channels, hidden_channels, heads=heads, concat=(i == num_layers - 1)))
            else:
                raise ValueError(f"Unknown encoder type: {enc_type}")

        final_embedding_dim = hidden_channels * heads if enc_type == 'GAT'else hidden_channels
        self.linear_mu = torch.nn.Linear(final_embedding_dim, out_channels)
        self.linear_logstd = torch.nn.Linear(final_embedding_dim, out_channels)

    def forward(self, x, edge_index, batch):
        # sequential forward pass
        for i in range(self.num_layers):
            x_input = x  # Store input for potential skip connection

            # Pass the output of the previous layer as input to the current one
            x = self.convs[i](x, edge_index)
            x_out = self.norm(x)
        
            if self.skip_connection and x_input.shape == x.shape:
                x = x_out + x_input

            else:
                # This can happen if the first layer changes the channel size.
                # In that case, you might skip the residual on the first layer or project x_input.
                x = x_out
                
            if i < self.num_layers - 1:
                x = self.norm(x)
                x = F.leaky_relu(x)

        # Global pooling happens after all layers are done
        x = global_mean_pool(x, batch)

        # Calculate mu and logstd from the final graph-level embedding
        mu = self.linear_mu(x)
        logstd = self.linear_logstd(x)

        return mu, logstd


class MLP_Decoder(torch.nn.Module):
    """
    Multi-Layer Perceptron (MLP) decoder for reconstructing dihedral angles
    from latent representations in a Variational Graph Autoencoder (VGAE).
    Parameters:
    - latent_dim (int): Dimensionality of the latent space (input features).
    - out_nodes (int): Number of nodes to reconstruct.
    - out_features (int): Number of features per node (e.g., 2 for dihedral angles).
    - hidden_channels (int): Number of hidden units in each MLP layer.
    - num_layers (int): Number of hidden layers in the MLP.
    Returns:
    - x_recon_radians (Tensor): Reconstructed dihedral angles in radians.
    """

    def __init__(self, latent_dim, out_nodes, out_features, hidden_channels, num_layers=3):
        super().__init__()
    
        self.out_nodes = out_nodes
        self.out_features = out_features
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.skip_connections = True

        # Define layers explicitly
        self.layers = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        
        # First layer
        self.layers.append(torch.nn.Linear(latent_dim, hidden_channels))
        self.activations.append(torch.nn.SiLU())

        # Hidden layers
        for i in range(num_layers - 1):
            self.layers.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.activations.append(torch.nn.SiLU())
        
        # Output layer
        self.final = torch.nn.Linear(hidden_channels, out_nodes * out_features * 2)

    def forward(self, z):
        x = z
        for i in range(self.num_layers):
            x_in = x
            x = self.layers[i](x)
            x = self.activations[i](x)

            # apply skip connection if dimensions match
            if self.skip_connections and x.shape == x_in.shape:
                x = x + x_in  

        logits = self.final(x)
        reshaped_logits = logits.view(-1, self.out_nodes, self.out_features, 2)

        x_coords = reshaped_logits[..., 0]
        y_coords = reshaped_logits[..., 1]

        # return angles in radians
        x_recon_radians = torch.atan2(y_coords, x_coords)
        return x_recon_radians



class DVGAE(torch.nn.Module):
    """
    Variational Graph Autoencoder (VGAE) for dihedral angle reconstruction.
    Combines a GNN-based encoder and an MLP-based decoder.
    Parameters:
    - encoder (torch.nn.Module): GNN encoder to map input graphs to latent space.
    - decoder (torch.nn.Module): MLP decoder to reconstruct dihedral angles from latent space.
    - device (torch.device): Device to run the model on (CPU or GPU).
    Returns:
    - x_recon (Tensor): Reconstructed dihedral angles.
    - mu (Tensor): Mean of the latent space distribution.
    - logstd (Tensor): Log standard deviation of the latent space distribution.
    """
    
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

    
