import torch
from torch import nn
from torch_geometric.nn.pool import global_mean_pool
from einops import rearrange, repeat

# Import the EGNN_Network from the file you provided
from updated_egnn_pytorch import EGNN_Network

# ==============================================================================
# HELPER FUNCTIONS: Bridge PyTorch Geometric and egnn-pytorch data formats
# ==============================================================================

def pyg_to_padded_feats(x, batch):
    """
    Converts PyG node features to padded batch format with a mask.
    """
    batch_size = batch.max().item() + 1
    num_nodes_per_graph = torch.bincount(batch)
    max_nodes = num_nodes_per_graph.max().item()

    padded_x = torch.zeros(batch_size, max_nodes, x.size(1), device=x.device)
    mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=x.device)

    for i in range(batch_size):
        node_indices = (batch == i)
        num_nodes = node_indices.sum()
        padded_x[i, :num_nodes] = x[node_indices]
        mask[i, :num_nodes] = True
        
    return padded_x, mask

def pyg_to_padded_edges(edge_index, edge_attr, batch):
    """
    Converts PyG edge data to the padded adjacency matrix format expected by the library.
    """
    batch_size = batch.max().item() + 1
    num_nodes_per_graph = torch.bincount(batch)
    max_nodes = num_nodes_per_graph.max().item()
    num_total_nodes, edge_dim = edge_attr.size()
    
    # This maps the global node index in the batch to a local index within its graph
    node_offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=edge_index.device)
    node_offsets[1:] = torch.cumsum(num_nodes_per_graph, dim=0)

    padded_edges = torch.zeros(batch_size, max_nodes, max_nodes, edge_dim, device=edge_attr.device)

    for i in range(batch_size):
        # Find edges belonging to the current graph
        edge_mask = (edge_index[0] >= node_offsets[i]) & (edge_index[0] < node_offsets[i+1])
        graph_edges = edge_index[:, edge_mask]
        graph_attrs = edge_attr[edge_mask]

        # Convert global indices to local indices for this graph
        local_edges = graph_edges - node_offsets[i]
        
        # Place edge attributes in the padded adjacency matrix
        padded_edges[i, local_edges[0], local_edges[1]] = graph_attrs

    return padded_edges

# ==============================================================================
# The VAE IMPLEMENTATION
# ==============================================================================

class Official_EGNN_Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_egnn_layers, latent_dim,
                mode= 'standard',
                pos_projection_dim =64,            
                noise_level=0.01,  
                attention=True,  # Optional attention mechanism            
                edge_attr=None, 
                pos_ref=None,
                ):
        

        super().__init__()
        self.latent_dim = latent_dim
        self.edge_attr = edge_attr
        self.pos_ref = pos_ref
        self.mode = mode
        self.noise_level = noise_level

        if self.mode not in ['standard', 'denoise']:
            raise ValueError("Mode must be either 'standard' or 'denoise'.")
        if self.mode == 'denoise':
            if noise_level <= 0:
                raise ValueError("Noise level must be positive for denoising mode.")
            print(f"Using denoising mode with noise level: {noise_level}")

        # Manually project 8-dim input features to the hidden dimension
        self.feature_project = nn.Linear(in_channels, hidden_channels)

        self.egnn_network = EGNN_Network(
            depth = num_egnn_layers,
            dim = hidden_channels,
            edge_dim = 2,  # <-- Correctly specify edge dimension
            #norm_feats = True, ALREADY DONE IN EGNN_Network
            norm_coors = True,
            fourier_features = 4,
            global_linear_attn_every= 1 if attention else 0,  # Use attention if specified
            inter_layer_norm_feats=True,
            inter_layer_norm_coors=False,      # often not needed; you can try True if coords drift/explode
            inter_layer_prenorm=True,
            inter_layer_postnorm=False
        )

        self.latent_project = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.LeakyReLU(),
        )
        

        self.to_latent_mean = nn.Linear(hidden_channels, latent_dim)
        self.to_latent_logvar = nn.Linear(hidden_channels, latent_dim)

    def forward(self, x, pos, edge_index, batch, edge_attr=None, new_noise_level=None):

        if new_noise_level is not None:
            if self.mode != 'denoise':
                raise ValueError("Cannot set noise level in non-denoising mode.")
            self.noise_level = new_noise_level
            

        if self.mode == 'denoise':
            pos = pos + torch.randn_like(pos) * self.noise_level  # Add small noise to positions
        # 1. Convert PyG data to padded format
        padded_x, mask = pyg_to_padded_feats(x, batch)
        padded_pos, _ = pyg_to_padded_feats(pos, batch)
        padded_edges = pyg_to_padded_edges(edge_index, edge_attr, batch)

        # 2. Project input features
        feats = self.feature_project(padded_x)

        # 3. Pass to the EGNN Network
        h_encoded, p_encoded = self.egnn_network(feats, padded_pos, edges=padded_edges, mask=mask)

        # 4. Pool using a masked mean
        mask_sum = mask.sum(-1, keepdim=True).clamp(min=1.0)
        graph_embedding = h_encoded.sum(dim=1) / mask_sum

        # critical to avoid exploding KL divergence
        #graph_embedding = self.latent_project(graph_embedding)

        
        # 5. Compute latent space parameters
        mean = self.to_latent_mean(graph_embedding)
        log_var = self.to_latent_logvar(graph_embedding)
        
        return mean, log_var

class Official_EGNN_Decoder(nn.Module):
    def __init__(self, latent_dim, node_feature_dim_initial, hidden_channels, num_egnn_layers,
                edge_attr=None,
                architecture='original', # or 'hybrid_displacement'
                attention=True,
                pos_ref=None,
                debug = False):
        super().__init__()

        self.architecture = architecture

        self.debug = debug

        self.egnn_network = EGNN_Network(
            depth=num_egnn_layers,
            dim=hidden_channels,
            edge_dim = 2, # edge_attr.shape[1] if edge_attr is not None else 0, # <-- Correctly specify edge dimension
            norm_coors = True,
            fourier_features = 4,
            global_linear_attn_every= 1 if attention else 0,  # Use attention if specified
            inter_layer_norm_feats=True,
            inter_layer_norm_coors=False,      # often not needed; you can try True if coords drift/explode
            inter_layer_prenorm=True,
            inter_layer_postnorm=False
        )
        
        # Layers to condition the network on the latent vector `z`
        self.map_latent = nn.Linear(latent_dim, hidden_channels)
        self.map_initial_features = nn.Linear(node_feature_dim_initial, hidden_channels)
        self.map_initial_pos = nn.Linear(hidden_channels, 3)

    def forward(self, z, x_initial_features, edge_index,batch, edge_attr = None,  pos_ref=None):
        # Convert PyG data to padded format
        padded_x, mask = pyg_to_padded_feats(x_initial_features, batch)
        padded_edges = pyg_to_padded_edges(edge_index, edge_attr, batch)

        # Create z-conditioned initial node features
        h_initial = self.map_initial_features(padded_x)
        z_proj = self.map_latent(z)
        z_expanded = repeat(z_proj, 'b d -> b n d', n=h_initial.size(1))
        h_conditioned = h_initial + z_expanded

        if self.architecture == 'hybrid_displacement':
            # For hybrid displacement, we need to condition on the reference positions
            if pos_ref is not None:
               pos_initial = pyg_to_padded_feats(pos_ref, batch)[0]
               #print(" hybrid pos_initial shape: ", pos_initial.shape)
            else:
                raise ValueError("pos_ref must be provided for hybrid displacement architecture.")
        else:
            # For the original architecture, we initialize positions from the conditioned features
            pos_initial = self.map_initial_pos(h_conditioned)
          

        # Pass to the EGNN to refine the structure
        h_decoded, pos_decoded = self.egnn_network(
            feats = h_conditioned, 
            coors = pos_initial, 
            edges = padded_edges,
            mask = mask
        )

        # Un-pad the final coordinates back to PyG format
        final_pos = pos_decoded[mask]
        
        return final_pos
