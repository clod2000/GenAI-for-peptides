import torch
from torch import nn
from torch_geometric.nn.pool import global_mean_pool
from einops import rearrange, repeat
import torch.nn.functional as F

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
                debug=False
             ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.edge_attr = edge_attr
        self.pos_ref = pos_ref
        self.mode = mode
        self.noise_level = noise_level

        self.debug = debug

        if self.mode not in ['standard', 'denoise']:
            raise ValueError("Mode must be either 'standard' or 'denoise'.")
        if self.mode == 'denoise':
            if noise_level <= 0:
                raise ValueError("Noise level must be positive for denoising mode.")
            print(f"Using denoising mode with noise level: {noise_level}")

        
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
            inter_layer_norm_coors=True,      # often not needed; you can try True if coords drift/explode
            inter_layer_prenorm=True,
            inter_layer_postnorm=False,
            m_pool_method = 'mean',
            coor_weights_clamp_value = 2.0 
        )
        
        self.combined_projection = nn.Sequential(
            nn.Linear(hidden_channels + 3, hidden_channels), # Project combined features
            nn.LayerNorm(hidden_channels),
            nn.SiLU()
        )

        self.to_latent_mean = nn.Linear(hidden_channels, latent_dim)
        self.to_latent_logvar = nn.Linear(hidden_channels, latent_dim)
           

        # Better initialization
        nn.init.xavier_normal_(self.to_latent_mean.weight, gain=0.01)
        nn.init.constant_(self.to_latent_mean.bias, 0.0)
        
        nn.init.xavier_normal_(self.to_latent_logvar.weight, gain=0.01)
        nn.init.constant_(self.to_latent_logvar.bias, -1.0)  # Start with smaller variance


    def forward(self, x, pos, edge_index, batch, edge_attr=None, new_noise_level=None):

        if new_noise_level is not None:
            if self.mode != 'denoise':
                raise ValueError("Cannot set noise level in non-denoising mode.")
            self.noise_level = new_noise_level
            

        if self.mode == 'denoise':
            pos = pos + torch.randn_like(pos) * self.noise_level  # Add small noise to positions


        # STAGE 1: Convert PyG data to padded format
        padded_x, mask = pyg_to_padded_feats(x, batch)
        padded_pos, _ = pyg_to_padded_feats(pos, batch)
        padded_edges = pyg_to_padded_edges(edge_index, edge_attr, batch)

        # STAGE 2: Project input features
        feats = self.feature_project(padded_x)

        # STAGE 3: Pass to the EGNN Network
        h_encoded, p_encoded = self.egnn_network(feats, padded_pos, edges=padded_edges, mask=mask)

        # STAGE 4: Create the Geometry-Aware Embedding, here I consider both features and position for the pooling
        #          build the latent space 

        # Pool the node features 
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        h_embedding = h_encoded.sum(dim=1) / mask_sum

        if self.debug:
            print(f"  -> H Embedding Shape: {h_embedding.shape}")
            print(f"  -> H Embedding Stats: min={h_embedding.min():.6f}, max={h_embedding.max():.6f}, mean={h_embedding.mean():.6f}")

        # Create a translation-invariant coordinate representation
        # I compute the mean position (center of mass) for each graph in the batch
        p_mean = p_encoded.sum(dim=1, keepdim=True)/  mask_sum.unsqueeze(-1)
        # Center the positions by subtracting the mean
        p_centered = p_encoded - p_mean
        # print(f"p_centered: {p_centered}")
        # print(f"p_mean: {p_mean}")
        # To combine, we can pool the centered positions as well.
        # Let's use mean pooling for the coordinates part of the embedding.
        p_embedding = p_centered.sum(dim=1) / mask_sum

        if self.debug:
            print(f"  -> P Embedding Shape: {p_embedding.shape}")
            print(f"  -> P Embedding Stats: min={p_embedding.min():.6f}, max={p_embedding.max():.6f}, mean={p_embedding.mean():.6f}")

        # Concatenate feature and geometry embeddings
        combined_embedding = torch.cat([h_embedding, p_embedding], dim=-1)

        if self.debug:
            print(f"  -> Combined Embedding Shape: {combined_embedding.shape}")
            print(f"  -> Combined Embedding Stats: min={combined_embedding.min():.6f}, max={combined_embedding.max():.6f}, mean={combined_embedding.mean():.6f}")
        
        # Project the combined embedding
        final_embedding = self.combined_projection(combined_embedding)

        final_embedding = F.normalize(final_embedding, p=2, dim=1)  # L2 normalization
        

        if self.debug:
            print(f"  -> Final Embedding Shape: {final_embedding.shape}")
            print(f"  -> Final Embedding Stats: min={final_embedding.min():.6f}, max={final_embedding.max():.6f}, mean={final_embedding.mean():.6f}")

        # 5. Compute latent space parameters from the richer embedding
        mean = self.to_latent_mean(final_embedding)
        log_var = self.to_latent_logvar(final_embedding)

        if self.debug:
            print(f"  -> Mean Shape: {mean.shape}, LogVar Shape: {log_var.shape}")
            print(f"  -> Mean Stats: min={mean.min():.6f}, max={mean.max():.6f}, mean={mean.mean():.6f}")
            print(f"  -> LogVar Stats: min={log_var.min():.6f}, max={log_var.max():.6f}, mean={log_var.mean():.6f}")
        
        return mean, log_var
    

class FiLMLayer(nn.Module):
    """A robust FiLM layer with proper initialization and gamma bounding."""
    def __init__(self, input_dim, condition_dim, debug=False):

        super().__init__()
        # It's often better to have separate linear layers for gamma and beta
        # to allow independent initialization and bounding.

        self.debug = debug

        self.gamma_generator = nn.Linear(condition_dim, input_dim)
        self.beta_generator = nn.Linear(condition_dim, input_dim)

        # Initialize gamma_generator to produce values that, when exponentiated, are close to 1.
        # A common way is to initialize weights to small values and bias to 0.
        # This makes exp(output) start near exp(0) = 1.
        nn.init.constant_(self.gamma_generator.weight, 0.) # Small weights
        nn.init.constant_(self.gamma_generator.bias, 0.)   # Bias to 0 for exp(0) = 1

        # Initialize beta_generator to produce values close to 0.
        nn.init.constant_(self.beta_generator.weight, 0.)
        nn.init.constant_(self.beta_generator.bias, 0.)

    def forward(self, x, z):
        # Generate raw gamma_logit and beta_raw from the latent vector z
        gamma_logit = self.gamma_generator(z)
        beta_raw = self.beta_generator(z)

        # Transform gamma_logit using exponential to ensure positive scale.
        # Clamp logit to prevent extreme exp() values.
        # (e.g., -5 to 5 maps exp to ~0.006 to 148, which is a good range)
        gamma = torch.exp(gamma_logit.clamp(min=-5., max=5.)) # Ensure gamma is positive and bounded

        if self.debug:
            print(f"  -> Gamma Logit Stats: min={gamma_logit.min():.6f}, max={gamma_logit.max():.6f}, mean={gamma_logit.mean():.6f}")
            print(f"  -> Gamma Stats: min={gamma.min():.6f}, max={gamma.max():.6f}, mean={gamma.mean():.6f}")

        # Beta (shift) is usually not bounded, but can be if prone to explosion
        # (e.g., beta = beta_raw.tanh() * max_beta_amplitude)
        beta = beta_raw # Typically left unbounded, relying on learning stability

        if self.debug:
            print(f"  -> Beta Raw Stats: min={beta_raw.min():.6f}, max={beta_raw.max():.6f}, mean={beta_raw.mean():.6f}")
            print(f"  -> Beta Stats: min={beta.min():.6f}, max={beta.max():.6f}, mean={beta.mean():.6f}")

        # Unsqueeze for broadcasting with node features (B, N, D)
        gamma = gamma.unsqueeze(1) # Shape: [B, 1, D]
        beta = beta.unsqueeze(1)   # Shape: [B, 1, D]
        
        return x * gamma + beta
    

class Official_EGNN_Decoder(nn.Module):
    def __init__(self, latent_dim, node_feature_dim_initial, hidden_channels, num_egnn_layers,
                edge_attr_dim=2,
                architecture='original', # or 'hybrid_displacement'
                attention=True,
                pos_ref=None,
                debug=False):
        super().__init__()

        self.architecture = architecture
        self.edge_attr_dim = edge_attr_dim
        self.debug = debug

        # --- Use FiLM for stronger conditioning ---
        self.film_conditioner = FiLMLayer(hidden_channels, latent_dim, debug=debug)
        
        self.map_initial_features = nn.Linear(node_feature_dim_initial, hidden_channels)
        
        # --- 1. A much more robust initial position generator ---
        self.map_initial_pos = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 3),
            nn.Tanh() # CRITICAL: Prevents coordinate collapse by bounding output
        )

        # --- 2. The main EGNN for refinement ---
        # Note: Consider a larger `depth` here for this architecture
        self.egnn_network = EGNN_Network(
            depth=num_egnn_layers,
            dim=hidden_channels,
            edge_dim=self.edge_attr_dim,  # Assuming your edge_attr has 2 features
            norm_coors=True,
            fourier_features=4,
            global_linear_attn_every=1 if attention else 0,
            inter_layer_norm_feats=True,
            inter_layer_norm_coors=True,
            inter_layer_prenorm=True,
            inter_layer_postnorm=False,
            m_pool_method = 'mean',
            coor_weights_clamp_value = 2.0 
        )

       
    def forward(self, z, x_initial_features, edge_index, batch, edge_attr=None, pos_ref=None):
        # Convert PyG data to padded format

    
        if self.debug:
            print(f"  -> Input z Shape: {z.shape}")
            print(f"  -> Initial Features Shape: {x_initial_features.shape}")
            print(f"  -> Edge Index Shape: {edge_index.shape}, Edge Attr Shape: {edge_attr.shape if edge_attr is not None else 'None'}")
            print(f"  -> Batch Shape: {batch.shape}")
           
        

        padded_x, mask = pyg_to_padded_feats(x_initial_features, batch)
        padded_edges = pyg_to_padded_edges(edge_index, edge_attr, batch)

        # Create z-conditioned initial node features using FiLM
        h_initial = self.map_initial_features(padded_x)
        h_conditioned = self.film_conditioner(h_initial, z)

        if self.debug:
            print(f"  -> Padded Initial Features Shape: {padded_x.shape}")
            print(f"  -> Conditioned Features Shape: {h_conditioned.shape}")
            print(f"  -> Mask Shape: {mask.shape}")

        # Generate a bounded, non-collapsed set of initial positions
        pos_initial = self.map_initial_pos(h_conditioned)

        # Pass to the EGNN to refine the structure from the initial guess
        h_decoded, pos_decoded = self.egnn_network(
            feats=h_conditioned, 
            coors=pos_initial, 
            edges=padded_edges,
            mask=mask
        )
    

        # Un-pad the final coordinates back to PyG format
        final_pos = pos_decoded[mask]
        
        return final_pos