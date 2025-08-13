import torch
import MDAnalysis as mda
from torch_geometric.data import Data, InMemoryDataset
import numpy as np

import torch_geometric as pyg
import torch._numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch

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

from torch_geometric.nn.pool import global_mean_pool, global_max_pool, global_add_pool


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SchNet, GATConv, global_mean_pool

from egnn_clean import EGNN



class EGNN_Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels_egnn=128, out_channels_egnn=128,
                 num_egnn_layers=4, latent_dim=64, num_nodes = 52,
                 architecture='original', # 'original' or 'hybrid_displacement'
                 pos_projection_dim=64,   # : For the hybrid model
                 mode = 'standard', # 'standard' or 'denoise' (for denoising autoencoder)
                 noise_level = 0.1, # For denoising autoencoder
                 edge_dim=None,
                 num_atom_types = None,
                 attention = False, 
                 tanh = False,
                 normalize = False,
                 verbose = False
                 ): # edge_dim if you have edge features
        
        """
        Initializes the EGNN Encoder.
        
        Args:   
            in_channels (int): Number of input features per node.
            hidden_channels_egnn (int): Number of hidden channels in EGNN layers.
            out_channels_egnn (int): Output dimension of the final EGNN layer.
            num_egnn_layers (int): Number of EGNN layers.
            latent_dim (int): Dimension of the latent space.
            num_nodes (int): Number of nodes in the graph (set during forward pass).
            architecture (str): Architecture type, either 'original' or 'hybrid_displacement'.
            pos_projection_dim (int): Dimension for position projection in hybrid architecture.
            mode (str): Mode of operation, either 'standard' or 'denoise'.
            noise_level (float): Standard deviation of Gaussian noise for denoising autoencoder.
            edge_dim (int, optional): Dimension of edge features if applicable.
            num_atom_types (int, optional): Number of atom types for one-hot encoding.
            attention (bool, optional): Whether to use attention mechanism in EGCL.
            tanh (bool, optional): Whether to use tanh activation in EGCL for coordinates MLP.
            verbose (bool, optional): If True, prints initialization details.
        """
        
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.in_channels = in_channels
        self.hidden_channels_egnn = hidden_channels_egnn
        self.out_channels_egnn = out_channels_egnn # Output dim of the final EGNN layer
        self.latent_dim = latent_dim
        self.num_egnn_layers = num_egnn_layers
        self.num_nodes = num_nodes # This will be set during the forward pass
        self.architecture = architecture
        self.pos_projection_dim = pos_projection_dim
        self.mode = mode
        self.noise_level = noise_level
        self.edge_dim = edge_dim


        self.project = nn.Linear(in_channels, hidden_channels_egnn) # Initial projection to hidden channels
        
        self.latent_projection = nn.Sequential(
            nn.Linear(out_channels_egnn, out_channels_egnn),
            nn.LayerNorm(out_channels_egnn),
            nn.LeakyReLU(),
        )
        self.egnn = EGNN(
            in_node_nf=in_channels,
            hidden_nf=hidden_channels_egnn,
            out_node_nf=out_channels_egnn,
            in_edge_nf=edge_dim if edge_dim is not None else 0,  # Edge features if provided
            device=self.device,
            n_layers=num_egnn_layers,
            tanh=tanh,  # Use tanh activation if specified
            attention=attention  # Use attention mechanism if specified
        ) 

        # Pooling layer to get graph-level embedding
        self.pool = global_mean_pool # Or global_add_pool, etc.

        if self.architecture == 'hybrid_displacement':
            # MLP that process the flattened, centered coordinates
            self.pos_processor = nn.Sequential(
                nn.Linear(3, pos_projection_dim // 2), # Project each coordinate
                nn.ReLU(),
                nn.Linear(pos_projection_dim // 2, pos_projection_dim)
            )
            # The final FC layers take input from both pooled features AND pooled positions
            self.fc_mean = nn.Linear(out_channels_egnn + pos_projection_dim, latent_dim)
            self.fc_log_var = nn.Linear(out_channels_egnn + pos_projection_dim, latent_dim)
        else: # Original architecture
            self.fc_mean = nn.Linear(out_channels_egnn, latent_dim)
            self.fc_log_var = nn.Linear(out_channels_egnn, latent_dim)
    
        if verbose:
            print(f"Encoder initialized with in_channels={in_channels}, hidden_channels_egnn={hidden_channels_egnn}, "
                  f"out_channels_egnn={out_channels_egnn}, num_egnn_layers={num_egnn_layers}, latent_dim={latent_dim}")
            print(f"Architecture: {architecture}")
            if architecture == 'hybrid_displacement':
                print(f"Position projection dimension: {pos_projection_dim}")
            print(f"Number of nodes: {num_nodes}")
            if edge_dim is not None:
                print(f"Edge features dimension: {edge_dim}")
            if num_atom_types is not None:
                print(f"Number of atom types: {num_atom_types}")
            if attention:
                print("Using attention mechanism in EGNN")
            if tanh:
                print("Using tanh activation in EGNN")

                


    def forward(self, x, pos, edge_index, batch, edge_attr=None, analyze=False):

        """
        Forward pass of the EGNN Encoder.
        Args:
            x (torch.Tensor): Node features of shape [num_nodes, in_channels].
            pos (torch.Tensor): Node positions of shape [num_nodes, 3].
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
            batch (torch.Tensor): Batch vector indicating the graph each node belongs to.
            analyze (bool): If True, returns intermediate representations for analysis.
            edge_attr (torch.Tensor, optional): Edge attributes of shape [num_edges, edge_dim].

        Returns:
            If analyze is True:
                h_enc (torch.Tensor): Encoded node features after EGNN layers.
                p_enc (torch.Tensor): Encoded node positions after EGNN layers.
                graph_embedding (torch.Tensor): Graph-level embedding after pooling.
                mean (torch.Tensor): Mean of the latent space distribution.
                log_var (torch.Tensor): Log variance of the latent space distribution.
            Else:   
                mean (torch.Tensor): Mean of the latent space distribution.
                log_var (torch.Tensor): Log variance of the latent space distribution.
        """
        
        h = x # Initial node features

        if self.mode == 'denoise':
            # Add Gaussian noise to the positions for denoising autoencoder
            p = pos + torch.randn_like(pos) * self.noise_level
        else:
            p = pos

        h_enc, p_enc = self.egnn(h, p, edges=edge_index, edge_attr=edge_attr)

        graph_embedding_h = self.pool(h_enc, batch)

        # --- ARCHITECTURE-SPECIFIC FORWARD PASS ---
        if self.architecture == 'hybrid_displacement':
            # 1. Center the coordinates to make them translation-invariant
            p_enc_centered = p_enc - self.pool(p_enc, batch).repeat_interleave(torch.bincount(batch), dim=0)

            # 2. Process each node's centered coordinate vector
            pos_features = self.pos_processor(p_enc_centered)

            # 3. Pool the coordinate features to get a fixed-size graph representation
            graph_embedding_p = self.pool(pos_features, batch)
            
            # 4. Concatenate feature embedding and position embedding
            final_graph_embedding = torch.cat([graph_embedding_h, graph_embedding_p], dim=1)
        else: # Original architecture
            final_graph_embedding = graph_embedding_h

        final_graph_embedding = self.latent_projection(final_graph_embedding)

        # Calculate latent space parameters from the final embedding
        mean = self.fc_mean(final_graph_embedding)
        log_var = self.fc_log_var(final_graph_embedding)

        if analyze:            
            return h_enc, p_enc, final_graph_embedding, mean, log_var
        else:
            return mean, log_var



class EGNN_Decoder(nn.Module):
    def __init__(self, latent_dim, node_feature_dim_initial, hidden_nf, num_egnn_layers, out_coord_dim=3,
                 pos_MLP_size= [256,128,128],
                 architecture='original', # 'original' or 'hybrid_displacement'
                 edge_dim=None, # Edge features dimension
                 attention=False, 
                 tanh=False,
                 normalize=False,
                 verbose=False,
                 ):
        
        """
        Initializes the EGNN Decoder.
        Args:   
            latent_dim (int): Dimension of the latent space.
            node_feature_dim_initial (int): Dimension of initial node features.
            hidden_nf (int): Number of hidden features in EGNN layers.
            num_egnn_layers (int): Number of EGNN layers.
            out_coord_dim (int, optional): Output dimension for positions (default is 3 for 3D coordinates).
            architecture (str): Architecture type, either 'original' or 'hybrid_displacement'.
            pos_MLP_size (list, optional): Sizes of the MLP layers for initial position prediction (default is [256, 128, 128]).
            edge_dim (int, optional): Dimension of edge features if applicable.
            attention (bool, optional): Whether to use attention mechanism in EGNN.
            tanh (bool, optional): Whether to use tanh activation in EGNN for coordinates MLP.
            normalize (bool, optional): Whether to normalize coordinates in EGNN.
            verbose (bool, optional): If True, prints initialization details.
        """
        
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.architecture = architecture
        self.edge_dim = edge_dim 
        self.latent_dim = latent_dim

        self.map_latent = nn.Linear(latent_dim, hidden_nf)  # Project latent vector to hidden dimension

        self.map_initial_node = nn.Linear(node_feature_dim_initial, hidden_nf)

        # --- ARCHITECTURE-SPECIFIC LAYERS ---
        if self.architecture == 'hybrid_displacement':
            # This EGNN is conditioned by the latent vector.

 
            self.egnn_decoder = EGNN(
                in_node_nf=hidden_nf, # Takes concatenated features
                hidden_nf=hidden_nf,
                out_node_nf=hidden_nf,
                n_layers=num_egnn_layers,
                in_edge_nf=self.edge_dim if self.edge_dim is not None else 0,  # Edge features if provided
                device=self.device,
                attention=attention,  # Use attention mechanism if specified
                tanh=tanh,  # Use tanh activation if specified
                normalize=normalize,  # Normalize coordinates if specified
                #node_attr_d=self.latent_dim  # Node attributes dimension
            )
            
        else: # Original architecture

            # self.initial_pos_MLP = nn.Sequential(
            #     nn.Linear(node_feature_dim_initial, pos_MLP_size[0]),
            #     nn.LeakyReLU(),
            #     nn.Linear(pos_MLP_size[0], pos_MLP_size[1]),
            #     nn.LeakyReLU(),
            #     nn.Linear(pos_MLP_size[1], pos_MLP_size[2]),
            #     nn.LeakyReLU(),
            #     nn.Linear(pos_MLP_size[2], out_coord_dim)  # Final output dimension for positions
            # )
            self.initial_pos_MLP = nn.Linear(hidden_nf, out_coord_dim)  # Initial position MLP

            self.egnn_decoder = EGNN(
                in_node_nf=node_feature_dim_initial,  # Input to EGNN layers
                hidden_nf=hidden_nf,
                out_node_nf=hidden_nf, # Output features from EGNN
                in_edge_nf=self.edge_dim if self.edge_dim is not None else 0,  # Edge features if provided
                n_layers=num_egnn_layers,
                device=self.device,
                attention=attention,  # Use attention mechanism if specified
                tanh=tanh,  # Use tanh activation if specified
                normalize=normalize,  # Normalize coordinates if specified
                #node_attr_d=self.latent_dim  # Node attributes dimension
            )

        if verbose:

            print(f"Decoder initialized with latent_dim={latent_dim}, node_feature_dim_initial={node_feature_dim_initial}, "
                  f"hidden_nf={hidden_nf}, num_egnn_layers={num_egnn_layers}, out_coord_dim={out_coord_dim}")
            print(f"Architecture: {architecture}")
            if architecture == 'hybrid_displacement':
                print(f"Latent vector will be projected to hidden dimension: {hidden_nf}")
            else: # Original architecture
                print(f"Initial position MLP sizes: {pos_MLP_size}")
        
            if attention:
                print("Using attention mechanism in EGNN")
            if tanh:
                print("Using tanh activation in EGNN for coordinates MLP")
            if normalize:
                print("Coordinates will be normalized in EGNN")

            print(f"Edge features dimension: {edge_dim}" if edge_dim is not None else "No edge features used in EGNN")
       

    def forward(self, z, x_initial_features, edge_index, batch, edge_attr=None, pos_ref=None, analyze=False):
        """
        Forward pass of the EGNN Decoder.
        Args:
            z (torch.Tensor): Latent vector of shape [num_nodes, latent_dim].
            x_initial_features (torch.Tensor): Initial node features of shape [num_nodes, node_feature_dim_initial].
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
            batch (torch.Tensor): Batch vector indicating the graph each node belongs to.
            pos_ref (torch.Tensor): Reference positions for the hybrid_displacement architecture.
            analyze (bool): If True, returns intermediate representations for analysis.
        Returns:
            If analyze is True:
                pos_decoded (torch.Tensor): Decoded node positions after EGNN layers.
                h_decoded (torch.Tensor): Decoded node features after EGNN layers.
                latent_pos (torch.Tensor): Initial positions from latent vector.
                h (torch.Tensor): Initial node features projected to EGNN's input feature dimension.
                z_repeated (torch.Tensor): Repeated latent vector for each node in the batch.
                pos_cat (torch.Tensor): Concatenated initial features and latent vector.
            Else:   
                pos_decoded (torch.Tensor): Decoded node positions after EGNN layers.
        """
        
        if self.architecture == 'hybrid_displacement':
            if pos_ref is None:
                raise ValueError("pos_ref must be provided for the hybrid_displacement decoder.")
            
            # We expand the single pos_ref to match the batch size.
            # num_graphs_in_batch = batch.max().item() + 1
            # # This assumes pos_ref is for a single graph.
            # if pos_ref.dim() == 1:
            #     pos_ref = pos_ref.repeat(num_graphs_in_batch, 1)
            # 1. Project initial features to the hidden dimension
            h = self.map_initial_node(x_initial_features)

            z_mapped = self.map_latent(z)
            z_conditioning = z_mapped[batch]
            h_conditioned = h + z_conditioning

            h_decoded, pos_decoded = self.egnn_decoder(h_conditioned, pos_ref, edge_index, edge_attr=edge_attr)

        else: # Original architecture

            z_repeated = z[batch]  # [N, latent_dim]

            h = self.map_initial_node(x_initial_features)  # Project to EGNN's input feature dimension

            z_features = self.map_latent(z_repeated)  # Project latent vector to hidden dimension

            h = h + z_features  # Condition initial features with latent vector
           
            #pos_cat = torch.cat([x_initial_features, z_repeated], dim=1) # [N, node_feature_dim_initial + latent_dim]
            
            pos_cat = x_initial_features  # [N, node_feature_dim_initial] - no latent vector concatenation in original architecture

             # the cat with initial features is to ensure that the initial position are at least slightly different for each node
            latent_pos = self.initial_pos_MLP(z_features+h) # Initial positions from latent vector [N, 3]
              
             # 1. Repeat z for every node
            #z_repeated = z[batch]

            # 2. Create base node features from input and from latent vector
            #h_initial = self.map_initial_node(x_initial_features)
            #z_features = self.map_latent(z_repeated)

            # 3. Condition the node features on z. This is the main information stream.
            #h = h_initial# + z_features

            # 4. Generate initial positions *directly from the conditioned features h*.
            #    This creates a clean, serial flow of information: z -> h -> pos
            #latent_pos = self.initial_pos_MLP(x_initial_features)
              
            # 5. Run the EGNN decoder, which refines the initial positions based on message passing
            #    that also uses the z-conditioned features 'h'.
            
            # Check if all positions are identical
            if torch.allclose(latent_pos, latent_pos[0:1].expand_as(latent_pos), atol=1e-6):
               print("WARNING: All generated positions are identical!")
        
            # Run EGNN decoder
            h_decoded, pos_decoded = self.egnn_decoder(x_initial_features, latent_pos, edge_index, edge_attr=edge_attr) #, node_attr=z_repeated)

        if analyze:
            return pos_decoded, h_decoded, h_conditioned, latent_pos, h, z_repeated
        else:
            return pos_decoded
    

class FGVAE(nn.Module):
    def __init__(self, encoder, decoder):
        """
        Initializes the FGVAE model with an encoder and decoder.
        Args:
            encoder (nn.Module): Encoder module.
            decoder (nn.Module): Decoder module.
        """
        super(FGVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean, log_var):
        """
        Reparameterization trick to sample from the latent space.
        Args:
            mean (torch.Tensor): Mean of the latent distribution.
            log_var (torch.Tensor): Log variance of the latent distribution.
        Returns:
            torch.Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, data, pos_ref=None, analyze=False):
        """
        Forward pass of the FGVAE model.
        Args:
            data (Data): Input data containing node features, edge indices, and batch information.
            pos_ref (torch.Tensor, optional): Reference positions for the hybrid_displacement architecture.
            analyze (bool): If True, returns intermediate representations for analysis.
        Returns:
            pos_pred (torch.Tensor): Predicted node positions.
            mean (torch.Tensor): Mean of the latent distribution.
            log_var (torch.Tensor): Log variance of the latent distribution.
            batch (torch.Tensor): Batch vector indicating the graph each node belongs to.
        """
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch
        edge_attr = data.edge_attr if 'edge_attr' in data else None
        
        if analyze:
            # Encode with analysis
            # chek if the option analyze is available in the encoder and decoder
            if not hasattr(self.encoder, 'analyze') or not hasattr(self.decoder, 'analyze'):
                raise ValueError("Encoder and Decoder must have 'analyze' option for analysis mode.")

            h_enc, p_enc, graph_embedding, mean, log_var = self.encoder(x, pos, edge_index, batch, edge_attr=edge_attr, analyze=True)
            # Reparameterize
            z = self.reparameterize(mean, log_var)
            # Decode with analysis
            pos_pred, h_decoded,h_conditioned, latent_pos, h_initial, z_repeated, pos_cat = self.decoder(z, x, edge_index, batch, edge_attr=edge_attr, pos_ref=pos_ref, analyze=True)
            return pos_pred, h_decoded, h_conditioned, latent_pos, h_initial, z_repeated, pos_cat, h_enc, p_enc, graph_embedding, mean, log_var, batch

        else:
            # Encode
            mean, log_var = self.encoder(x, pos, edge_index, batch= batch, edge_attr=edge_attr)

            # Reparameterize
            #z = self.reparameterize(mean, log_var)
            z = mean # TEMPORARY CHECK

            # Decode
            pos_pred = self.decoder(z, x, edge_index, pos_ref=pos_ref, batch=batch, edge_attr=edge_attr)

            return pos_pred, mean, log_var, batch

    def generate(self, data_sample, pos_ref=None, z=None):
        """
        Generate new conformations from the prior distribution or using a specific latent vector.
        Args:
            data_sample (Data): Sample data to use for generating new conformations.
            pos_ref (torch.Tensor, optional): Reference positions for the hybrid_displacement architecture.
            z (torch.Tensor, optional): Specific latent vector to use for generation. If None, samples from standard normal distribution.
        Returns:
            pos_pred (torch.Tensor): Predicted node positions.
        """
        if z is None:
            # Sample from standard normal distribution
            z = torch.randn(1, self.encoder.latent_dim, device=data_sample.x.device)

        if z.shape[1] != self.encoder.latent_dim:
            raise ValueError(f"Latent vector must have shape [batch_size, {self.encoder.latent_dim}]")
        
        # Decode using the decoder
        pos_pred = self.decoder(z, data_sample.x, data_sample.edge_index, data_sample.batch)

        return pos_pred
    
