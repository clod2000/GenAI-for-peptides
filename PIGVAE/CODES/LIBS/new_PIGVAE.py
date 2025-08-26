import torch
import MDAnalysis as mda
from torch_geometric.data import Data, InMemoryDataset

import torch_geometric as pyg
# Using standard numpy is recommended over torch._numpy
import numpy as np
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

        self.num_pairs = (num_nodes * (num_nodes - 1)) // 2  # Number of unique pairs for distance embedding


        self.feature_project = nn.Linear(in_channels, hidden_channels_egnn) # Initial projection to hidden channels

        self.latent_projection = nn.Sequential(
            nn.Linear(2*out_channels_egnn, out_channels_egnn),
            nn.LayerNorm(out_channels_egnn),
            nn.SiLU(),
            nn.Linear(out_channels_egnn, out_channels_egnn)
        )


        self.egnn = EGNN(
            in_node_nf=hidden_channels_egnn,
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

        pos_projection_dim = self.out_channels_egnn

        self.pos_processor = nn.Sequential(
            nn.Linear(self.num_pairs, min(1024, self.num_pairs)), # Project each coordinate
            nn.LayerNorm(min(1024, self.num_pairs)),
            nn.SiLU(),
            nn.Linear(min(1024, self.num_pairs), min(512, self.num_pairs//2)),
            nn.LayerNorm(min(512, self.num_pairs//2)),
            nn.SiLU(),
            nn.Linear(min(512, self.num_pairs//2), min(256, self.num_pairs//4)),
            nn.LayerNorm(min(256, self.num_pairs//4)),
            nn.SiLU(),
            nn.Linear(min(256, self.num_pairs//4), pos_projection_dim),
            nn.LayerNorm(pos_projection_dim)
        )

        self.to_latent_mean = nn.Linear(out_channels_egnn, latent_dim)
        self.to_latent_logvar = nn.Linear(out_channels_egnn, latent_dim)

        nn.init.xavier_normal_(self.to_latent_mean.weight, gain=1.0)  # Standard gain
        nn.init.zeros_(self.to_latent_mean.bias)
        nn.init.xavier_normal_(self.to_latent_logvar.weight, gain=1.0)
        nn.init.zeros_(self.to_latent_logvar.bias)  # Start with neutral variance

        if verbose:
            print(f"Encoder initialized with in_channels={in_channels}, hidden_channels_egnn={hidden_channels_egnn}, "
                  f"out_channels_egnn={out_channels_egnn}, num_egnn_layers={num_egnn_layers}, latent_dim={latent_dim}")
            # ... (other print statements are fine) ...

    def forward(self, x, pos, edge_index, batch, edge_attr=None, analyze=False):
        if self.mode == 'denoise':
            pos = pos + torch.randn_like(pos) * self.noise_level

        h = self.feature_project(x)
        h_enc, p_enc = self.egnn(h, pos, edges=edge_index, edge_attr=edge_attr)
        graph_embedding_h = self.pool(h_enc, batch)

        batch_size = batch.max().item() + 1 if batch is not None else 1
        distance_embeddings = []

        for i in range(batch_size):
            mask = batch == i
            pos_i = p_enc[mask]
            dist_matrix = torch.cdist(pos_i, pos_i)
            triu_indices = torch.triu_indices(dist_matrix.size(0), dist_matrix.size(1), offset=1)
            distances = dist_matrix[triu_indices[0], triu_indices[1]]
            distance_embeddings.append(distances)

        distance_embeddings = torch.stack(distance_embeddings)
        graph_embedding_p = self.pos_processor(distance_embeddings)

        final_graph_embedding = torch.cat([graph_embedding_h, graph_embedding_p], dim=-1)
        final_graph_embedding = self.latent_projection(final_graph_embedding)

        mean = self.to_latent_mean(final_graph_embedding)
        log_var = self.to_latent_logvar(final_graph_embedding)

        if analyze:
            return h_enc, p_enc, final_graph_embedding, mean, log_var
        else:
            return mean, log_var

class FiLMLayer(nn.Module):
    """A robust FiLM layer with proper initialization and gamma bounding."""
    def __init__(self, input_dim, condition_dim, debug=False):
        super().__init__()
        self.debug = debug
        self.gamma_generator = nn.Linear(condition_dim, input_dim)
        self.beta_generator = nn.Linear(condition_dim, input_dim)
        nn.init.constant_(self.gamma_generator.weight, 0.)
        nn.init.constant_(self.gamma_generator.bias, 0.)
        nn.init.constant_(self.beta_generator.weight, 0.)
        nn.init.constant_(self.beta_generator.bias, 0.)

    def forward(self, x, z):
        gamma_logit = self.gamma_generator(z)
        beta_raw = self.beta_generator(z)
        gamma = torch.exp(gamma_logit.clamp(min=-5., max=5.))
        beta = beta_raw
        return x * gamma + beta

class EGNN_Decoder(nn.Module):
    def __init__(self, latent_dim, node_feature_dim_initial, hidden_nf, num_egnn_layers, out_coord_dim=3,
                 pos_MLP_size= [256,128,128],
                 architecture='original',
                 edge_dim=None,
                 attention=False,
                 tanh=False,
                 normalize=False,
                 verbose=False,
                 ):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.architecture = architecture
        self.edge_dim = edge_dim
        self.latent_dim = latent_dim
        hidden_channels = hidden_nf

        # ### FIX START ###
        # CHANGE 1: REMOVE THE LAYER THAT PROCESSES LEAKED FEATURES
        # This layer processed the ground-truth `data.x`, causing the information leak.
        # self.map_initial_features = nn.Linear(node_feature_dim_initial, hidden_channels)

        # CHANGE 2: ADD A LAYER TO GENERATE FEATURES FROM THE LATENT VECTOR
        # This is the new, correct starting point for the decoder's feature pipeline.
        self.map_latent_to_features = nn.Linear(self.latent_dim, hidden_channels)
        # ### FIX END ###

        # The rest of the __init__ method is fine
        self.film_conditioner = FiLMLayer(hidden_channels, latent_dim)

        self.map_initial_pos = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 3),
            nn.Tanh() # CRITICAL: Prevents coordinate collapse by bounding output
        )

        self.egnn_decoder = EGNN(
            in_node_nf=hidden_channels,
            hidden_nf=hidden_nf,
            out_node_nf=hidden_nf,
            n_layers=num_egnn_layers,
            in_edge_nf=self.edge_dim if self.edge_dim is not None else 0,
            device=self.device,
            attention=attention,
            tanh=tanh,
            normalize=normalize,
        )

        if verbose:
            print(f"Decoder initialized with latent_dim={latent_dim}, node_feature_dim_initial={node_feature_dim_initial}, "
                  f"hidden_nf={hidden_nf}, num_egnn_layers={num_egnn_layers}, out_coord_dim={out_coord_dim}")
            # ... (other print statements are fine) ...

    # ### FIX START ###
    # CHANGE 3: UPDATE THE FORWARD SIGNATURE
    # `x_initial_features` is removed from the arguments to prevent the leak.
    def forward(self, z, edge_index, batch, edge_attr=None, pos_ref=None, analyze=False):
    # ### FIX END ###
        
        # Expand the graph-level latent vector `z` to have one vector per node
        z_repeated = z[batch]  # Shape: [num_total_nodes, latent_dim]

        # ### FIX START ###
        # CHANGE 4: APPLY THE NEW FEATURE GENERATION LOGIC
        # Generate the initial node features directly from the latent vector.
        h_initial = self.map_latent_to_features(z_repeated)
        # ### FIX END ###
        
        # The rest of the pipeline now correctly builds upon information
        # that originated *only* from z.
        h_conditioned = self.film_conditioner(h_initial, z_repeated)

        if self.architecture == 'hybrid_displacement':
            if pos_ref is None:
                raise ValueError("pos_ref must be provided for the hybrid_displacement decoder.")
            # In hybrid mode, we use the provided reference coordinates as the starting point
            pos_initial = pos_ref
        else: # Original architecture
            # In original mode, we generate the initial coordinates from the conditioned features
            pos_initial = self.map_initial_pos(h_conditioned)

        # Run the main EGNN decoder to refine features and positions
        h_decoded, pos_decoded = self.egnn_decoder(h_conditioned, pos_initial, edge_index, edge_attr=edge_attr)

        if analyze:
            # Adjust return values as needed since some variables no longer exist
            return pos_decoded, h_decoded, h_conditioned, pos_initial
        else:
            return pos_decoded


class FGVAE(nn.Module):
    def __init__(self, encoder, decoder, AE = False):
        super(FGVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.AE = AE

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, data, pos_ref=None, analyze=False, new_noise_level=None):
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch
        edge_attr = data.edge_attr if 'edge_attr' in data else None
        
        if analyze:
            # (Analysis mode would also need to be updated based on the decoder's new return signature)
            raise NotImplementedError("Analysis mode needs to be updated for the corrected decoder.")
        
        # 1. Encode the input data to get the latent distribution
        mean, log_var = self.encoder(x, pos, edge_index, batch= batch, edge_attr=edge_attr)
        log_var = log_var.clamp(min=-10, max=10)

        # 2. Sample from the latent distribution
        if self.AE:
            z = mean
        else:
            z = self.reparameterize(mean, log_var)
       
        # ### FIX START ###
        # CHANGE 5: CALL THE DECODER WITHOUT THE LEAKED `x` TENSOR
        # The decoder now only receives the latent vector `z` and graph topology info.
        pos_pred = self.decoder(z, edge_index, batch, edge_attr=edge_attr, pos_ref=pos_ref)
        # ### FIX END ###

        return pos_pred, mean, log_var, batch

    def generate(self, data_sample, pos_ref=None, z=None):
        # We need a device for the new z tensor
        device = data_sample.x.device if hasattr(data_sample, 'x') else 'cpu'

        if z is None:
            # Sample from standard normal distribution
            z = torch.randn(1, self.encoder.latent_dim, device=device)

        if z.shape[1] != self.encoder.latent_dim:
            raise ValueError(f"Latent vector must have shape [batch_size, {self.encoder.latent_dim}]")
        
        # Ensure batch tensor is correctly set for a single sample
        batch = torch.zeros(data_sample.num_nodes, dtype=torch.long, device=device)
        edge_attr = data_sample.edge_attr if hasattr(data_sample, 'edge_attr') else None

        # ### FIX START ###
        # CHANGE 6: CALL THE DECODER WITHOUT THE LEAKED `x` TENSOR
        pos_pred = self.decoder(z, data_sample.edge_index, batch, edge_attr=edge_attr, pos_ref=pos_ref)
        # ### FIX END ###

        return pos_pred