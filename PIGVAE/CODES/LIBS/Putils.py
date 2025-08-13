import torch
import MDAnalysis as mda
from torch_geometric.data import Data, InMemoryDataset
import numpy as np
import os
import os.path as osp
import warnings
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
import pandas as pd
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.nn.pool import global_mean_pool
import math
from torch.utils.tensorboard import SummaryWriter
import itertools
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import summary, VGAE
from tqdm import tqdm

from create_full_graph_data_physics import TrajectoryDataset

import sys

# === ADDED IMPORTS FOR ANALYSIS ===
import seaborn as sns
from sklearn.decomposition import PCA
import imageio
import io

############################## data loading and preprocessing functions ##############################

def get_dataset(root_dir = None,
                tpr_file = 'MD.tpr',
                trajectory = 'MD_with_solvent_noPBC.xtc',
                selection = 'protein',
                include_atom_type = True,
                scale_features = True,
                scale_pos = True,
                initial_alignment = False,
                verbose = True,
                return_max_position = False,
                return_pos_angstrom = True
                 ):

    """
    Function to load a dataset from a given root directory, TPR file, and trajectory file.
    It preprocesses the data by scaling positions, optionally including atom types,
    scaling features, and aligning the first frame to the origin.

    NOTE: - The dataset is designed to have the same features for all graphs, so the first graph is used to extract the features.
          - The positions are scaled to have a maximum value of 1 if `scale_pos` is set to True.
          - The features are scaled to have zero mean and unit variance if `scale_features` is set to True.
          - If `initial_alignment` is set to True, the first frame is aligned to the origin and all other frames are aligned to the first frame.
    Args:
        root_dir (str): The root directory where the dataset is stored. If None, uses a default path.
        tpr_file (str): The TPR file containing the topology information.
        trajectory (str): The trajectory file containing the atomic positions.
        selection (str): The selection string to filter the atoms in the trajectory.
        include_atom_type (bool): Whether to include atom type features in the dataset.
        scale_features (bool): Whether to scale the features to have zero mean and unit variance.
        scale_pos (bool): Whether to scale the positions to a maximum value of 1.
        initial_alignment (bool): Whether to align the first frame to the origin and all other frames to the first frame.
        verbose (bool): Whether to print verbose output during processing.
        return_max_position (bool): Whether to return the maximum position value used for scaling.
        return_pos_angstrom (bool): Whether the positions should be returned in Angstroms (True) or nm (False).
        dataset (InMemoryDataset): The processed dataset containing the atomic positions and features.
    """ 

    
    if verbose: print("Loading dataset ...")
    if verbose: print(f"Root directory: {root_dir}")
    if verbose: print(f"TPR file: {tpr_file}")
    if verbose: print(f"Trajectory file: {trajectory}")
    if verbose: print(f"Selection: {selection}")
    if verbose: print(f"Include atom type: {include_atom_type}")
    if verbose: print(f"Scale features: {scale_features}")
    if verbose: print(f"Scale positions: {scale_pos}")
    if verbose: print(f"Initial alignment: {initial_alignment}")
    if verbose: print(f"Return max position: {return_max_position}")
    if verbose: print(f"Return position in Angstrom: {return_pos_angstrom}")

    if root_dir is None:
        if verbose: print("No root directory provided, using default path ...")
        root_dir = osp.join(osp.dirname(__file__), '..', '..','DATA',)

    
    # This will load the preprocessed .pt file if it exists, or create it if not.
    dataset = TrajectoryDataset(root=root_dir,
                                tpr_filename=tpr_file,
                                trajectory_filename=trajectory,
                                selection=selection)

    if len(dataset) == 0:
        print("Dataset is empty. Exiting.")
        exit()


    positions = dataset.pos

    if  not return_pos_angstrom:
        if verbose: print("Converting positions to nm ...")
        # Convert positions from Angstroms to nm
        positions = positions / 10.0
    else:
        if verbose: print("Positions are already in Angstroms, no conversion needed ...")
    
    # Get the maximum value of position for scaling
    if scale_pos:
        # define the positions tensor to find the maximum for each axis
        #posit = torch.cat([data.pos for data in dataset], dim=0)
        max_position = torch.max(np.fabs(positions), dim=0).values
        if verbose: print(f"\nScaling positions... Max absolute position values for scaling: {max_position}")
        
        positions = positions/max_position
            
    positions = positions.view(-1,dataset[0].num_nodes,3)  # Reshape to (num_graphs, num_nodes, 3)
    if verbose: print(f"Positions shape: {positions.shape}")

    features = dataset[0].x  # Assuming all graphs have the same features
    
    
    if include_atom_type:

        if verbose: print("Including atom features: performing one hot encoding ...")
        # Step 1: Extract the first column (categorical feature)
        first_column = features[:, 0].long()  # convert to integer indices
        # Step 2: Get the unique values (categories)
        unique_values = torch.unique(first_column)
        num_classes = len(unique_values)
        # Optional: Map the unique values to a continuous index space (e.g., 6→0, 1→1, etc.)
        value_to_index = {val.item(): idx for idx, val in enumerate(unique_values)}
        indexed_column = torch.tensor([value_to_index[x.item()] for x in first_column])
        # Step 3: One-hot encode
        one_hot_encoded = F.one_hot(indexed_column, num_classes=num_classes).float()

        if not scale_features:
            features = torch.cat((one_hot_encoded,features[:,1:]),dim=1)
    
    else:
        if verbose: print("Not including atom features, discarding it ...")
        features = features[:,1:] # Discard it anyway because if include_atom_type the next step will add it again

    if scale_features:

        if verbose: print(f"Scaling features ...")

        # Scale the features to have zero mean and unit variance
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features[:,1:].numpy())
        features = torch.tensor(scaled_features, dtype=torch.float32)
        if include_atom_type:
            # If atom type is included, concatenate the one-hot encoded features
            features = torch.cat((one_hot_encoded, torch.tensor(scaled_features, dtype=torch.float32)), dim=1)


    new_dataset = [] # will hold the new dataset

    if initial_alignment:
        if verbose: print("Aligning the first frame to the origin ...")
        # Align the first frame to the origin
        first_frame = positions[0]
        center_of_mass = torch.mean(first_frame, dim=0)
        aligned_first_frame = first_frame - center_of_mass
        dataset[0].pos = aligned_first_frame
        
        if verbose: print("Aligning all the other frame to the first frame ...")

        for pos,data in zip(positions,dataset):   # here I keep also the first even if already aligned for simplicity                
            R,t = find_rigid_alignment(pos, aligned_first_frame)
            aligned_pos = torch.matmul(pos, R.T) + t

            new_data = Data(x=features, edge_index=data.edge_index, pos=aligned_pos, batch=data.batch)  
            new_dataset.append(new_data)
    else:
        if verbose: print("not aligning the frames ...")

        for pos,data in zip(positions,dataset): 
            new_data = Data(x=features, edge_index=data.edge_index, pos=pos, batch=data.batch)
            new_dataset.append(new_data)            
    
    # convert the dataset to a PyTorch Geometric dataset

    dataset = InMemoryDataset(root=root_dir, transform=None)
    dataset.data, dataset.slices = dataset.collate(new_dataset)
    dataset.num_graphs = len(dataset)
    if verbose: 
        print(f"Dataset created with {dataset.num_graphs} graphs.")
        print(f"Number of graphs in the dataset: {dataset.num_graphs}")
        print(f"Number of features in the dataset: {dataset[0].num_features}")
        print(f"Number of edges in the dataset: {dataset[0].edge_index.size(1)}")
        print(f"Number of nodes in the dataset: {dataset[0].num_nodes}")
        print(f"Number of features in the dataset: {dataset[0].x.shape}")
        print()
    
    if return_max_position:
        if verbose: print(f"Returning max position value: {max_position}")
        return dataset, max_position
    
    return dataset

def create_physics_informed_dataset(dataset: InMemoryDataset, physics_critic) -> InMemoryDataset:
    """
    Enriches a PyTorch Geometric dataset with physics-based features.

    This function performs two main operations:
    1.  Appends Lennard-Jones parameters (sigma, epsilon) to the node features (`data.x`).
    2.  Creates bond parameters (r0, k_bond) as new edge attributes (`data.edge_attr`).

    Args:
        dataset (InMemoryDataset): The input dataset to be enriched. Assumes all graphs
                                   share the same topology and node feature matrix before enrichment.
        physics_critic (EnergyCalculator): An instantiated EnergyCalculator object that has
                                           already parsed the PDB/topology and contains the
                                           bond and Lennard-Jones parameters.

    Returns:
        InMemoryDataset: A new dataset where each Data object contains the updated
                         node features and new edge attributes.
    """
    print("Creating a physics-informed dataset...")

    # === Step 1: Prepare Node-Level Lennard-Jones Features ===
    print("  - Preparing Lennard-Jones node features (sigma, epsilon)...")
    sigma_list = physics_critic.sigma_list
    epsilon_list = physics_critic.epsilon_list
    lj_features = torch.tensor(
        list(zip(sigma_list, epsilon_list)),
        dtype=torch.float32
    )

    # Assume original features are constant across the dataset
    original_features = dataset[0].x
    # New feature matrix by concatenating original features with LJ parameters
    new_node_features = torch.cat([original_features, lj_features], dim=1)

    # === Step 2: Prepare Edge-Level Bond Features ===
    print("  - Preparing bond edge attributes (r0, k_bond)...")
    # Create an efficient lookup dictionary: { (atom1, atom2): [r0, k] }
    bond_params_lookup = {}
    for i, bond_indices in enumerate(physics_critic.bonds):
        p1, p2 = bond_indices
        key = tuple(sorted((p1, p2)))  # Canonical key for undirected edges
        r0 = physics_critic.r0_list[i]
        k = physics_critic.k_list[i]
        bond_params_lookup[key] = [r0, k]

    # Create the edge_attr tensor. Assumes constant topology.
    # We only need to do this once.
    edge_index = dataset[0].edge_index
    edge_attributes_list = []
    for j in range(edge_index.size(1)):
        u, v = edge_index[:, j].tolist()
        key = tuple(sorted((u, v)))
        # Default to [0, 0] if a graph edge is not in our physics topology (e.g., H-bonds)
        params = bond_params_lookup.get(key, [0.0, 0.0])
        edge_attributes_list.append(params)
    
    new_edge_attributes = torch.tensor(edge_attributes_list, dtype=torch.float32)

    # === Step 3: Build the New Dataset ===
    print("  - Assembling the new dataset...")
    new_data_list = []
    for data in tqdm(dataset, desc="Updating graphs"):
        # Create a new Data object with the enriched features.
        # This is safer than modifying the original data objects in place.
        new_data = Data(
            x=new_node_features,
            edge_index=data.edge_index,
            pos=data.pos,
            edge_attr=new_edge_attributes # All graphs share this
        )
        # Copy over any other attributes that might exist
        for key, value in data:
            if key not in ['x', 'edge_index', 'pos', 'edge_attr']:
                new_data[key] = value
        
        new_data_list.append(new_data)

    # === Step 4: Collate into a final InMemoryDataset ===
    final_dataset = InMemoryDataset(root=dataset.root, transform=None)
    final_dataset.data, final_dataset.slices = final_dataset.collate(new_data_list)
    
    print("Physics-informed dataset created successfully!")
    print(f"  - New node feature dimension: {final_dataset[0].num_features}")
    print(f"  - New edge attribute dimension: {final_dataset[0].edge_attr.shape[1]}")
    
    return final_dataset


def add_physics_attributes_to_dataset(dataset, physics_critic):
    """
    Adds bond parameters (r0, k) as edge attributes to each graph in the dataset.

    This function iterates through the dataset, and for each graph, it creates
    an `edge_attr` tensor where each row corresponds to an edge in `edge_index`
    and contains the physical parameters for that bond.
    """
    print("Adding physics parameters as edge attributes...")

    # Step 1: Create a lookup dictionary for efficient parameter access.
    # The key is a sorted tuple of atom indices for a bond.
    bond_params_lookup = {}
    for i, bond_indices in enumerate(physics_critic.bonds):
        p1, p2 = bond_indices
        key = tuple(sorted((p1, p2))) # Use a sorted tuple to handle undirected edges
        r0 = physics_critic.r0_list[i]
        k = physics_critic.k_list[i]
        bond_params_lookup[key] = [r0, k]

    # Step 2: Iterate through each graph and build its edge_attr tensor.
    new_data_list = []
    for i, data in enumerate(tqdm(dataset, desc="Processing graphs")):
        edge_attributes = []
        for j in range(data.edge_index.size(1)):
            u, v = data.edge_index[:, j].tolist()
            key = tuple(sorted((u, v)))

            # Get the parameters for this specific edge.
            # If an edge exists in the graph but not the topology, use defaults.
            params = bond_params_lookup.get(key, [0.0, 0.0])
            edge_attributes.append(params)

        # Create the final tensor and add it to the data object.
        data.edge_attr = torch.tensor(edge_attributes, dtype=torch.float32)
        new_data_list.append(data)

    # Re-collate the dataset with the new attribute
    new_dataset = InMemoryDataset(root=dataset.root, transform=None)
    new_dataset.data, new_dataset.slices = new_dataset.collate(new_data_list)
    
    print(f"Finished. Edge attributes added with shape: {new_dataset[0].edge_attr.shape}")
    return new_dataset



def find_rigid_alignment(source,target, check_reflection=True):

    A = source
    B = target
    """
    Aligns predicted and true positions using Kabsch algorithm.
    The Kabsch algorithm finds the optimal rotation and translation
    that minimizes the root mean square deviation (RMSD) between two sets of points.
    The algorithm assumes that the two sets of points are in the same coordinate system.
    The algorithm works by centering the points, computing the covariance matrix,
    performing singular value decomposition (SVD) on the covariance matrix,
    and then computing the optimal rotation and translation.
    
        Args:
        -    A: Torch tensor of shape (N,D) -- Point Cloud to Align (source)
        -    B: Torch tensor of shape (N,D) -- Reference Point Cloud (target)
        -   check_reflection: bool -- If True, checks for reflection and corrects the rotation matrix if necessary.
        Returns:
        -    R: optimal rotation
        -    t: optimal translation
    
    """

    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.mm(B_c)
    U, S, V = torch.svd(H) # V here is the original V from SVD

    # Calculate initial R
    R = V.mm(U.T)

    # Check for reflection and correct if necessary
    if check_reflection:
        # If the determinant of R is negative, it indicates a reflection
        # We can correct this by flipping the last column of V
        # This ensures that the rotation matrix R has a positive determinant
        if torch.det(R) < 0:
            # print("Reflection detected, correcting R...") # Optional debug print
            V_prime = V.clone()  # Work on a copy of V for the modification
            V_prime[:, -1] *= -1
            R = V_prime.mm(U.T)  # Recompute R with the modified copy

    # Translation vector
    # Ensure R used for translation is the potentially corrected one
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T 
    t = t.T
    return R, t.squeeze()


def get_dataloaders(dataset, batch_size=32, shuffle=True, num_workers=0, seed=42, train_ratio=0.7, val_ratio=0.2, verbose = False):
    """
    Function to create train, validation and test dataloaders from a dataset.
    
    Args:
        dataset (InMemoryDataset): The dataset to split into train, validation and test sets.
        batch_size (int): The batch size for the dataloaders.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of workers for data loading.
        seed (int): Random seed for reproducibility.
        split_ratio (float): Ratio to split the dataset into train, validation and test sets.

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        test_loader (DataLoader): DataLoader for the test set.
    """
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)

    if verbose: print("\nCreating dataloaders ...")
    
    # Calculate the number of samples for train and test sets
    num_samples = len(dataset)
    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    test_size = num_samples - train_size - val_size
    if test_size < 0:
        raise ValueError("The sum of train and validation ratios exceeds 1. Please adjust the ratios.")
    if val_size < 0:
        raise ValueError("The validation ratio is too high. Please adjust the ratio.")
    if train_size < 0:
        raise ValueError("The train ratio is too high. Please adjust the ratio.")
    # Split the dataset into train, validation and test sets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed))
    if len(train_dataset) == 0:
        raise ValueError("The training dataset is empty. Please check the dataset and the split ratios.")
    if len(val_dataset) == 0:
        raise ValueError("The validation dataset is empty. Please check the dataset and the split ratios.")
    if len(test_dataset) == 0:
        raise ValueError("The test dataset is empty. Please check the dataset and the split ratios.")
    # Create DataLoaders for train, validation and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers)
    if len(train_loader) == 0:
        raise ValueError("The training DataLoader is empty. Please check the dataset and the batch size.")
    if len(val_loader) == 0:
        raise ValueError("The validation DataLoader is empty. Please check the dataset and the batch size.")
    if len(test_loader) == 0:
        raise ValueError("The test DataLoader is empty. Please check the dataset and the batch size.")  
    if verbose:
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        print(f"Number of features in the dataset: {dataset[0].num_features}")
        print(f"Number of edges in the dataset: {dataset[0].edge_index.size(1)}")
        print(f"Number of nodes in the dataset: {dataset[0].num_nodes}")
        print(f"Number of features in the dataset: {dataset[0].x.shape}")
        print()
  

    return train_loader, val_loader, test_loader



def parse_config(config_file,verbose=False):
    """
    Parse configuration file and return a dictionary of parameters.
    The configuration file should be in the format:
    key: value
    where key is the parameter name and value is the parameter value.
    
    Args:
        config_file (str): Path to the configuration file.
    Returns:
        config (dict): Dictionary containing the parameters from the configuration file.
    """
    config = {}

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
    if not config_file.endswith('.in'):
        raise ValueError(f"Configuration file '{config_file}' should have a '.in' extension.")
    
    if verbose: print(f"Parsing configuration file: {config_file}")
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('//') or line.startswith('#'):
                #if verbose: print(f"Skipping line: {line}")
                continue
                
            # Parse key-value pairs
            if ':' in line or '=' in line:
                key, value = line.split(':', 1) if ':' in line else line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Convert values to appropriate types
                if value.lower() == 'none':
                    config[key] = None
                elif value.lower() in ['true', 'false']:
                    config[key] = value.lower() == 'true'
                else:
                    try:
                        # Try to convert to int first
                        config[key] = int(value)
                    except ValueError:
                        try:
                            # Try to convert to float
                            config[key] = float(value)
                        except ValueError:
                            # Handle list/array values (e.g., [256,256,128])
                            if value.startswith('[') and value.endswith(']'):
                                # Remove brackets and split by comma
                                list_str = value[1:-1].strip()
                                if list_str:  # Check if not empty
                                    try:
                                        config[key] = [int(x.strip()) for x in list_str.split(',')]
                                    except ValueError:
                                        try:
                                            config[key] = [float(x.strip()) for x in list_str.split(',')]
                                        except ValueError:
                                            config[key] = [x.strip() for x in list_str.split(',')]
                                else:
                                    config[key] = []
                            else:
                                # Keep as string
                                config[key] = value
    
    return config



###################################### LOSS FUNCTIONS ######################################

def KL_divergence(mu, logvar):
    """
    Compute the KL divergence between the learned distribution and the prior distribution.
    
    Args:
        mu (torch.Tensor): Mean of the learned distribution.
        logvar (torch.Tensor): Log variance of the learned distribution.
    
    Returns:
        torch.Tensor: KL divergence value.
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def reconstruction_loss(pos_pred, pos_true, batch, align=True):
    """
    Compute the reconstruction loss between predicted and true positions.
    The loss is computed as the mean squared error (MSE) between the predicted and true positions.
    If `align` is True, the predicted positions are aligned to the true positions using the Kabsch algorithm.
    Args:
        pos_pred (torch.Tensor): Predicted positions of shape (N, 3).
        pos_true (torch.Tensor): True positions of shape (N, 3).
        batch (torch.Tensor): Batch indices of shape (N,) indicating which graph each point belongs to.
        align (bool): Whether to align the predicted positions to the true positions using Kabsch algorithm.
    Returns:
        torch.Tensor: Mean squared error loss value.
    """


    total_loss = 0.0
    num_graphs = batch.max().item() + 1 # Get number of graphs in the batch

    for i in range(num_graphs):
        # Extract points for the current graph
        pred_mask = (batch == i)
        true_mask = (batch == i) # Assuming batch is the same for pred/true if generated correctly

        current_pos_pred = pos_pred[pred_mask]
        current_pos_true = pos_true[true_mask]

        # Ensure there are points to align (might happen with filtering/padding)
        if current_pos_pred.shape[0] == 0 or current_pos_true.shape[0] == 0:
            continue
            
        # Check if number of points match (should always match for VAE reconstruction)
        if current_pos_pred.shape[0] != current_pos_true.shape[0]:
            raise ValueError(f"Shape mismatch for graph {i} in batch: "
                            f"Pred {current_pos_pred.shape}, True {current_pos_true.shape}")

        if align:
            # Align the predicted points to the true points
            # Using Kabsch algorithm to find optimal rotation and translation
            # This function is defined above
            R, t = find_rigid_alignment(current_pos_pred, current_pos_true)
            # Apply the transformation to the predicted points
            current_pos_pred = (R @ current_pos_pred.T).T + t
       
        # Calculate MSE loss for this graph
        loss_i = F.mse_loss(current_pos_pred, current_pos_true, reduction='mean')
        total_loss += loss_i

    # Average loss over the graphs in the batch
    return total_loss / num_graphs if num_graphs > 0 else torch.tensor(0.0, device=pos_pred.device)



## More advanced loss functions that preserve molecular geometry better than simple MSE

def coordinate_loss(pos_pred, pos_true, batch, align=True):
    """
    Standard coordinate-wise loss with optional alignment
    """
    if not align:
        return F.mse_loss(pos_pred, pos_true)
    
    total_loss = 0.0
    num_graphs = batch.max().item() + 1
    
    for i in range(num_graphs):
        mask = (batch == i)
        pos_pred_i = pos_pred[mask]
        pos_true_i = pos_true[mask]
        
        # Align predicted to true
        R, t = find_rigid_alignment(pos_pred_i, pos_true_i)
        pos_pred_aligned = (R @ pos_pred_i.T).T + t
        
        # Calculate MSE after alignment
        graph_loss = F.mse_loss(pos_pred_aligned, pos_true_i)
        total_loss += graph_loss
    
    return total_loss / num_graphs

def distance_matrix_loss(pos_pred, pos_true, batch):
    """
    Distance matrix loss - inherently alignment-invariant
    No explicit alignment needed
    """
    total_loss = 0.0
    num_graphs = batch.max().item() + 1
    
    for i in range(num_graphs):
        mask = (batch == i)
        pos_pred_i = pos_pred[mask]
        pos_true_i = pos_true[mask]
        
        # Calculate pairwise distances - invariant to rotation/translation
        d_pred = torch.cdist(pos_pred_i, pos_pred_i)
        d_true = torch.cdist(pos_true_i, pos_true_i)
        
        # Upper triangular part only (avoid redundancy)
        triu_indices = torch.triu_indices(d_pred.size(0), d_pred.size(0), offset=1)
        d_pred_triu = d_pred[triu_indices[0], triu_indices[1]]
        d_true_triu = d_true[triu_indices[0], triu_indices[1]]
        
        graph_loss = F.mse_loss(d_pred_triu, d_true_triu)
        total_loss += graph_loss
        
    return total_loss / num_graphs

def bond_angle_loss(pos_pred, pos_true, edge_index):
    """
    Bond and angle loss - inherently alignment-invariant
    No explicit alignment needed
    """
    # Bond length component - invariant to rotation/translation
    sender, receiver = edge_index
    true_bonds = pos_true[sender] - pos_true[receiver]
    pred_bonds = pos_pred[sender] - pos_pred[receiver]
    true_lengths = torch.norm(true_bonds, dim=1)
    pred_lengths = torch.norm(pred_bonds, dim=1)
    bond_loss = F.mse_loss(pred_lengths, true_lengths)
    
    # Bond angle component - also invariant to rotation/translation
    # [Implementation as before]
    
    return bond_loss  # + angle_loss

def advanced_reconstruction_loss(pos_pred, pos_true, edge_index, batch, align_coords=True):
    """
    Comprehensive loss function that properly handles alignment requirements
    """
    # 1. Coordinate loss - needs explicit alignment if requested
    coord_loss = coordinate_loss(pos_pred, pos_true, batch, align=align_coords)
    
    # 2. Distance matrix loss - inherently alignment-invariant
    dist_loss = distance_matrix_loss(pos_pred, pos_true, batch)
    
    # 3. Bond/angle loss - inherently alignment-invariant
    structure_loss = bond_angle_loss(pos_pred, pos_true, edge_index)
    
    # Combine with weights
    return 0.2 * coord_loss + 0.5 * dist_loss + 0.3 * structure_loss




#################################### model functions ####################################

def print_model_summary(model):
    """
    Print a summary of the model including the number of parameters and trainable parameters.
    Args:
        model (torch.nn.Module): The model to summarize.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("The model should be an instance of torch.nn.Module")
        
    print("="*50)
    print("MODEL SUMMARY")
    print("="*50)
    
    total_params = 0
    trainable_params = 0
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            num_params = sum(p.numel() for p in module.parameters())
            num_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            if num_params > 0:
                print(f"{name:50} | {str(module):60} | {num_params:>10,} | {num_trainable:>10,}")
                total_params += num_params
                trainable_params += num_trainable
    
    print("="*50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("="*50)


def beta_annealer(epochs,beta_start = 0., beta_end = 1., annealing_epochs = 100, wait_epochs = 10):
    
    """
    Compute the beta value for the current epoch.
    If the current epoch is less than the wait_epochs, return the beta_start value.
    Otherwise, linearly interpolate between beta_start and beta_end based on the current epoch.
    """

    if epochs < wait_epochs:
        return beta_start
    
    return beta_start + (beta_end - beta_start) * min(1,((epochs-wait_epochs) / annealing_epochs))

# def lambda_annealer(epochs, lambda_start = 0., lambda_end = 1., annealing_epochs = 100, wait_epochs = 10):
#     """
#     Compute the lambda value for the current epoch.
#     If the current epoch is less than the wait_epochs, return the lambda_start value.
#     Otherwise, linearly interpolate between lambda_start and lambda_end based on the current epoch.
#     """
    
#     if epochs < wait_epochs:
#         return lambda_start
    
#     return lambda_start + (lambda_end - lambda_start) * min(1,((epochs-wait_epochs) / annealing_epochs))










############################# Visualization functions #############################


#### Function used in notebook_lr_layers

# function used to plot a dataset of graphs in 3D, used to check alignment and scaling of the graphs
def plot_graph_dataset(dataset, n_graphs, ax = None, title='Graph'):

    palette = ["r", "g", "b", "y", "c", "m", "k"]
    k=0
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        k=1
    for i in range(n_graphs):
       
        if i >= len(dataset):
            print(f"Graph index {i} out of range. Only {len(data)} graphs available.")
            return
        
        data = dataset[i]


        G = to_networkx(data, to_undirected=True)
        pos = data.pos.numpy()
    
        # Draw nodes
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c= palette[i % len(palette)], s=50, label=f'graph_{i+1}')
        
        # Draw edges
        for edge in G.edges():
            x = [pos[edge[0], 0], pos[edge[1], 0]]
            y = [pos[edge[0], 1], pos[edge[1], 1]]
            z = [pos[edge[0], 2], pos[edge[1], 2]]
            ax.plot(x, y, z, color = palette[i % len(palette)], alpha=0.5, linewidth=1)
        
    ax.set_title(title)
    if k == 1:
        ax.legend()
        plt.tight_layout()
        plt.show()
    else:
        ax.legend()
        plt.tight_layout()
        return ax

     
        
def plot_graph_pred(pos, true_pos_graph, ax=None, title='Graph', planewise=False, quiver=True):
    """
    Function to plot the predicted graph positions and the true graph positions.
    If `planewise` is True, it will plot the graph in three different planes (XY, XZ, YZ).
    If `ax` is None, it will create a new figure and axes for the plot. 
    If `ax` is provided, it will use the provided axes for plotting.
    Args:
        pos (torch.Tensor): Predicted positions of the graph nodes of shape (N, 3).
        true_pos_graph (Data): True graph data containing the true positions and edge indices.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure and axes will be created.
        title (str, optional): Title of the plot. Defaults to 'Graph'.
        planewise (bool, optional): If True, plot in three different planes (XY, XZ, YZ). Defaults to False.
        quiver (bool, optional): If True, add displacement vectors. Defaults to True.
    Returns:
        ax (matplotlib.axes.Axes): The axes with the plotted graph.
    """
    if not isinstance(true_pos_graph, Data):
        raise TypeError("true_pos_graph should be an instance of torch_geometric.data.Data")
    
    palette = ["r", "g", "b", "y", "c", "m", "k"]
    
    G = to_networkx(true_pos_graph, to_undirected=True)
    pos_true = true_pos_graph.pos.numpy()

    graph_pred = true_pos_graph.clone()
    graph_pred.pos = pos
    G_pred = to_networkx(graph_pred, to_undirected=True)

    pos_pred = graph_pred.pos.numpy()

    k = 0  # Flag to determine if we should show the plot
    
    if planewise:
        if ax is not None:
            raise ValueError("When planewise is True, ax should be None to create subplots.")
        
        # Create subplots for planewise plotting
        fig = plt.figure(figsize=(15, 5))
        ax_names = ['X-Y Plane', 'X-Z Plane', 'Y-Z Plane']
        ax_names_labels = ['X-axis', 'Y-axis', 'Z-axis']
        
        for i in range(3):
            current_ax = fig.add_subplot(1, 3, i + 1)
            current_ax.set_title(ax_names[i])
            current_ax.set_xlabel(ax_names_labels[i % 3])
            current_ax.set_ylabel(ax_names_labels[(i + 1) % 3])

            if i == 0:
                pos_true_plane = pos_true[:, [0, 1]]
                pos_pred_plane = pos_pred[:, [0, 1]]
            elif i == 1:
                pos_true_plane = pos_true[:, [0, 2]]
                pos_pred_plane = pos_pred[:, [0, 2]]
            else:
                pos_true_plane = pos_true[:, [1, 2]]
                pos_pred_plane = pos_pred[:, [1, 2]]

            # Draw nodes
            current_ax.scatter(pos_true_plane[:, 0], pos_true_plane[:, 1], color=palette[0], s=50, label='graph_true')
            current_ax.scatter(pos_pred_plane[:, 0], pos_pred_plane[:, 1], color=palette[1], s=50, label='graph_pred')

            # Draw edges for true positions
            for edge in G.edges():
                x = [pos_true_plane[edge[0], 0], pos_true_plane[edge[1], 0]]
                y = [pos_true_plane[edge[0], 1], pos_true_plane[edge[1], 1]]
                current_ax.plot(x, y, color=palette[0], alpha=0.5, linewidth=1)

            # Draw edges for predicted positions
            for edge in G_pred.edges():
                x = [pos_pred_plane[edge[0], 0], pos_pred_plane[edge[1], 0]]
                y = [pos_pred_plane[edge[0], 1], pos_pred_plane[edge[1], 1]]
                current_ax.plot(x, y, color=palette[1], alpha=0.5, linewidth=1)

            # Add quiver arrows if requested
            if quiver:
                current_ax.quiver(pos_true_plane[:, 0], pos_true_plane[:, 1],
                                pos_pred_plane[:, 0] - pos_true_plane[:, 0],
                                pos_pred_plane[:, 1] - pos_true_plane[:, 1],
                                color='k', alpha=0.7, width=0.003, 
                                angles='xy', scale_units='xy', scale=1,
                                label='Displacement Vectors' if i == 0 else "")

            current_ax.grid(True)
            current_ax.legend()
        
        plt.tight_layout()
        plt.show()
        return None
    
    else:
        # 3D plotting
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            k = 1  # We created the figure, so we should show it
        
        ax.set_title(title)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        
        try:
            ax.set_zlabel('Z-axis')
        except Exception as e:
            warnings.warn(f"Could not set Z-axis label: {e}. This might be due to the current Axes not being 3D.")

        # Draw nodes
        ax.scatter(pos_true[:, 0], pos_true[:, 1], pos_true[:, 2], color=palette[0], s=50, label='graph_true')
        ax.scatter(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2], color=palette[1], s=50, label='graph_pred')

        # Draw edges for true positions
        for edge in G.edges():
            x = [pos_true[edge[0], 0], pos_true[edge[1], 0]]
            y = [pos_true[edge[0], 1], pos_true[edge[1], 1]]
            z = [pos_true[edge[0], 2], pos_true[edge[1], 2]]
            ax.plot(x, y, z, alpha=0.5, color=palette[0], linewidth=1)

        # Draw edges for predicted positions
        for edge in G_pred.edges():
            x = [pos_pred[edge[0], 0], pos_pred[edge[1], 0]]
            y = [pos_pred[edge[0], 1], pos_pred[edge[1], 1]]
            z = [pos_pred[edge[0], 2], pos_pred[edge[1], 2]]
            ax.plot(x, y, z, alpha=0.5, color=palette[1], linewidth=1)

        # Add quiver arrows if requested
        if quiver: 
            ax.quiver(pos_true[:, 0], pos_true[:, 1], pos_true[:, 2],
                    pos_pred[:, 0] - pos_true[:, 0],
                    pos_pred[:, 1] - pos_true[:, 1],
                    pos_pred[:, 2] - pos_true[:, 2],
                    color='k', alpha=0.4, linewidth=2, arrow_length_ratio=0.05, 
                    label='Displacement Vectors')

        ax.legend()
        
        if k == 1:
            plt.tight_layout()
            plt.show()
        else:
            plt.tight_layout()
            return ax



#################################### ANALYSIS FUNCTIONS ########################################

def save_trajectory_as_pdb(coords_list, ref_pdb_path, out_path,verbose=True):
    """Saves a list of coordinate sets as a multi-model PDB file.
    Assumes coordinates are in Angstroms."""
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    
    # Load reference universe for topology
    u = mda.Universe(ref_pdb_path)
    
    with mda.Writer(out_path, n_atoms=u.atoms.n_atoms) as W:
        for i, coords in enumerate(coords_list):
            coords_np = coords.detach().cpu().numpy().reshape(-1, 3)
            u.atoms.positions = coords_np
            u.trajectory.ts.frame = i
            W.write(u.atoms)
    if verbose: print(f"Saved {len(coords_list)} frames to {out_path}")

def analyze_reconstruction_quality(model, test_loader, device, file_path, physics_critic, pos_ref, scale_pos, max_pos, ref_pdb, num_samples=10, verbose=True):
    """Analyzes reconstruction by finding best/worst RMSD, plotting, and saving PDBs."""
    model.eval()
    all_samples = []
    
    for i, data in enumerate(tqdm(test_loader, desc="Analyzing Reconstructions", leave=False)):
        if i >= num_samples: break
        
        data = data.to(device)
        with torch.no_grad():
            pos_pred_scaled, _, _, _ = model(data, pos_ref=pos_ref)
        
            if scale_pos:
                pos_pred_nm = pos_pred_scaled * max_pos
                pos_true_nm = data.pos * max_pos
            else:
                pos_pred_nm = pos_pred_scaled
                pos_true_nm = data.pos

            # Align for RMSD calculation
            R, t = find_rigid_alignment(pos_pred_nm.squeeze(0), pos_true_nm.squeeze(0))
            pos_pred_nm_aligned = (R @ pos_pred_nm.squeeze(0).T).T + t
            rmsd = torch.sqrt(torch.mean(torch.sum((pos_pred_nm_aligned - pos_true_nm.squeeze(0))**2, dim=1))).item()

            # Energy calculation
            pred_energy = physics_critic.openMM_energy(pos_pred_nm.squeeze(0)).item()
            true_energy = physics_critic.openMM_energy(pos_true_nm.squeeze(0)).item()
            
            all_samples.append({
                'true_coords_nm': pos_true_nm.squeeze(0),
                'pred_coords_nm': pos_pred_nm.squeeze(0),
                'aligned_pred_coords_nm': pos_pred_nm_aligned,
                'true_energy': true_energy,
                'pred_energy': pred_energy,
                'rmsd': rmsd,
                'id': i
            })

    # Sort by RMSD to find best and worst
    all_samples.sort(key=lambda x: x['rmsd'])
    best_sample = all_samples[0]
    worst_sample = all_samples[-1]
    
    # Save PDBs (convert to Angstrom)
    try: 
        save_trajectory_as_pdb([s['true_coords_nm'] * 10 for s in [best_sample, worst_sample]], ref_pdb, f"{file_path}/reconstruction_true_best_worst.pdb")
    except Exception as e:
        print(f"Error saving true coordinates PDB: {e}")
    try:
        save_trajectory_as_pdb([s['pred_coords_nm'] * 10 for s in [best_sample, worst_sample]], ref_pdb, f"{file_path}/reconstruction_pred_best_worst.pdb")
    except Exception as e:
        print(f"Error saving predicted coordinates PDB: {e}")

    # Plot best reconstruction
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    true_np = best_sample['true_coords_nm'].cpu().numpy()
    aligned_pred_np = best_sample['aligned_pred_coords_nm'].cpu().numpy()
    
    ax.scatter(true_np[:, 0], true_np[:, 1], true_np[:, 2], c='b', label='True', alpha=0.5, s=20)
    ax.scatter(aligned_pred_np[:, 0], aligned_pred_np[:, 1], aligned_pred_np[:, 2], c='r', label='Predicted (Aligned)', alpha=0.5, s=20)
    ax.set_title(f"Best Reconstruction (RMSD: {best_sample['rmsd']:.3f} nm)\nE_true: {best_sample['true_energy']:.1f} | E_pred: {best_sample['pred_energy']:.1f} (kJ/mol)")
    plt.legend()
    plt.savefig(f"{file_path}/reconstruction_best.png")
    plt.close()

def analyze_latent_space(model, dataloader, device, file_path, physics_critic, scale_pos, max_pos, verbose=True):
    """Encodes the dataset, performs PCA, and plots the latent space colored by energy."""
    model.eval()
    latent_vectors = []
    energies = []
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Analyzing Latent Space", leave=False):
            data = data.to(device)
            edge_att = data.edge_attr if hasattr(data, 'edge_attr') else None
            mean, _ = model.encoder(data.x, data.pos, data.edge_index, data.batch, edge_attr=edge_att)
            latent_vectors.append(mean.cpu())
            
            if scale_pos:
                true_coords_nm = data.pos * max_pos
            else:
                true_coords_nm = data.pos
            
            energy_val = physics_critic.openMM_energy(true_coords_nm.squeeze(0))
            energies.append(energy_val.item())

    latent_vectors = torch.cat(latent_vectors, dim=0).numpy()
    energies = np.array(energies)

    # Use PCA to reduce to 2D
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_vectors)

    plt.figure(figsize=(10, 8))
    points = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=energies, cmap='viridis', s=20, alpha=0.7)
    plt.colorbar(points, label='Potential Energy (kJ/mol)')
    plt.title('Latent Space Visualization of Test Set (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"{file_path}/latent_space_pca.png")
    plt.close()

def analyze_generation_and_energy(model, test_loader, device, file_path, physics_critic, pos_ref, scale_pos, max_pos, ref_pdb, n_generate=100, verbose=True):
    """Generates new structures and compares their energy distribution to the real data."""
    model.eval()
    
    # 1. Get energy distribution of the real test set
    real_energies = []
    data_sample = None
    for data in tqdm(test_loader, desc="Calculating Real Energies", leave=False):
        if data_sample is None: data_sample = data.to(device)
        data = data.to(device)
        
        if scale_pos:
            true_coords_nm = data.pos * max_pos
        else:
            true_coords_nm = data.pos
        energy_val = physics_critic.openMM_energy(true_coords_nm.squeeze(0))
        real_energies.append(energy_val.item())

    # 2. Generate new samples
    if data_sample is None:
        print("Test loader is empty, cannot perform generation analysis.")
        return
    generated_energies = []
    generated_coords_list = []
    with torch.no_grad():
        for _ in tqdm(range(n_generate), desc="Generating New Structures", leave=False):
            z = torch.randn(1, model.encoder.latent_dim, device=device)
            edge_att = data_sample.edge_attr if hasattr(data_sample, 'edge_attr') else None
            pos_pred_scaled = model.decoder(z, data_sample.x, data_sample.edge_index, data_sample.batch, pos_ref=pos_ref, edge_attr=edge_att)

            if scale_pos:
                generated_coords_nm = pos_pred_scaled * max_pos
            else:
                generated_coords_nm = pos_pred_scaled
            
            generated_coords_list.append(generated_coords_nm.squeeze(0))
            energy_val = physics_critic.openMM_energy(generated_coords_nm.squeeze(0))
            generated_energies.append(energy_val.item())

    # 3. Plot the energy distributions
    plt.figure(figsize=(10, 6))
    sns.histplot(real_energies, color="blue", label="Real Test Data", kde=True, stat="density", bins=30)
    sns.histplot(generated_energies, color="red", label="Generated Samples", kde=True, stat="density", bins=30)
    plt.title("Potential Energy Distribution Comparison")
    plt.xlabel("Energy (kJ/mol)")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(f"{file_path}/energy_distribution.png")
    plt.close()
    
    # 4. Save the 5 lowest energy generated structures (in Angstrom)
    sorted_indices = np.argsort(generated_energies)
    low_energy_coords_angstrom = [generated_coords_list[i] * 10 for i in sorted_indices[:5]]
    save_trajectory_as_pdb(low_energy_coords_angstrom, ref_pdb, f"{file_path}/generated_low_energy.pdb")

def analyze_interpolation(model, test_loader, device, file_path, pos_ref, ref_pdb_path, scale_pos, max_pos, n_steps=30,verbose=True):
    """Interpolates between two latent space points and saves the resulting trajectory."""
    model.eval()
    if len(test_loader.dataset) < 2:
        print("Need at least two samples in test set for interpolation.")
        return
        
    data1 = test_loader.dataset[0].to(device)
    data2 = test_loader.dataset[1].to(device)


    if test_loader.batch_size == 1:
        batch = torch.tensor([0] * len(data1.pos), dtype=torch.long, device=device)  # Assuming all data belongs to batch 0
    else:
        batch = data1.batch

    edge_att_1 = data1.edge_attr if hasattr(data1, 'edge_attr') else None
    edge_att_2 = data2.edge_attr if hasattr(data2, 'edge_attr') else None

    interp_coords_angstrom = []
    with torch.no_grad():
        z1, _ = model.encoder(data1.x, data1.pos, data1.edge_index, batch, edge_attr=edge_att_1)
        z2, _ = model.encoder(data2.x, data2.pos, data2.edge_index, batch, edge_attr=edge_att_2)

        z_interp = [torch.lerp(z1, z2, t) for t in np.linspace(0, 1, n_steps)]


        for z in tqdm(z_interp, desc="Analyzing Interpolation", leave=False):
            pos_pred_scaled = model.decoder(z, data1.x, data1.edge_index, batch, pos_ref=pos_ref, edge_attr=edge_att_1)
            if scale_pos:
                pos_pred_nm = pos_pred_scaled * max_pos
            else:
                pos_pred_nm = pos_pred_scaled
            interp_coords_angstrom.append(pos_pred_nm.squeeze(0) * 10) # to Angstrom
            
    # Save as multi-model PDB
    save_trajectory_as_pdb(interp_coords_angstrom, ref_pdb_path, f"{file_path}/interpolation_path.pdb")

    

    # Create and save a GIF
    images = []
    for i, coords_a in enumerate(interp_coords_angstrom):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        coords_np = coords_a.cpu().numpy()
        ax.scatter(coords_np[:, 0], coords_np[:, 1], coords_np[:, 2], c=coords_np[:, 2], cmap='viridis', alpha=0.8)
        ax.set_title(f"Interpolation Step {i+1}/{n_steps}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Save the figure to an in-memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=fig.dpi) # Use savefig to a buffer
        buf.seek(0) # Rewind the buffer to the beginning
        
        # Read the image from the buffer using imageio
        image = imageio.imread(buf)
        images.append(image)
        
        # Clean up the buffer and figure to save memory
        buf.close()
        plt.close(fig)

        plt.close(fig)
    
    imageio.mimsave(f"{file_path}/interpolation_path.gif", images, fps=5)

def run_full_analysis(model, test_loader, device, file_path, config, physics_critic, pos_ref, max_positions, epoch, verbose = True):
    """Runs a full suite of analysis on the trained model."""
    print("\n" + "="*20 + " RUNNING FULL ANALYSIS " + "="*20)
    analysis_path = os.path.join(file_path, f'analysis_results_{epoch}')
    if not os.path.exists(analysis_path): os.makedirs(analysis_path)

    scale_pos = config.get('SCALE_POSITIONS', True)
    ref_pdb = config.get('PDB_FOR_ENERGY')
    
    if not config.get('USE_FORCE_FIELD', True) or physics_critic is None:
        if verbose: print("Skipping analysis that requires a force field (energy, etc.).")
    else:
        if verbose: print("1. Analyzing reconstruction quality...")
        try:
            analyze_reconstruction_quality(model, test_loader, device, analysis_path, physics_critic, pos_ref, scale_pos, max_positions, ref_pdb, num_samples=len(test_loader), verbose=verbose)
        except Exception as e:
            print(f"Error occurred during reconstruction quality analysis: {e}")

        if verbose: print("\n2. Analyzing latent space...")
        try:
            analyze_latent_space(model, test_loader, device, analysis_path, physics_critic, scale_pos, max_positions, verbose=verbose)
        except Exception as e:
            print(f"Error occurred during latent space analysis: {e}")

        if verbose: print("\n3. Analyzing generation and energy distribution...")
        try:
            analyze_generation_and_energy(model, test_loader, device, analysis_path, physics_critic, pos_ref, scale_pos, max_positions, ref_pdb, n_generate=len(test_loader.dataset), verbose=verbose)
        except Exception as e:
            print(f"Error occurred during generation and energy distribution analysis: {e}")

    if verbose: print("\n4. Analyzing latent space interpolation...")
    try:
        analyze_interpolation(model, test_loader, device, analysis_path, pos_ref, ref_pdb, scale_pos, max_positions, n_steps=30, verbose=verbose)
    except Exception as e:
        print(f"Error occurred during latent space interpolation analysis: {e}")

    if verbose: print("\n" + "="*20 + " ANALYSIS COMPLETE " + "="*20)
    if verbose: print(f"Results saved in: {analysis_path}")



######### to be revised and understood #########

# def compute_tc_vae_loss(x_pred, x_true, edge_index, mean, logvar, batch, beta=1.0, tc_weight=1.0):
#     """
#     Compute the β-TCVAE loss with decomposed KL terms
    
#     Args:
#         x_pred: Reconstructed data
#         x_true: Original data  
#         mean: Encoder mean output
#         logvar: Encoder log variance output
#         batch: Batch indices
#         beta: Overall KL weight (like your existing BETA parameter)
#         tc_weight: Weight for total correlation term
#     """
#     # Reconstruction loss (same as before)
#     recon_loss = advanced_reconstruction_loss(x_pred, x_true, edge_index, batch)

#     # Standard KL - ensure this is calculated properly
#     kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
    
#     # Calculate decomposed KL terms
#     # Instead of complex MC estimation, use simple approximation for stability
#     batch_size = mean.size(0)
#     latent_dim = mean.size(1)
    
#     # 1. Total correlation approximation (how much dims depend on each other)
#     z = reparameterize(mean, logvar)
#     log_qz_mean = gaussian_log_density(z, mean, logvar)  # [B]

#     # Mean across batch dimension for each latent dim
#     mean_mean = mean.mean(dim=0, keepdim=True)  # [1, D]
#     mean_logvar = torch.log(torch.exp(logvar).mean(dim=0, keepdim=True))  # [1, D]
    
#     # Compute log q(z) - approximate marginalization  
#     log_qz_marginal = gaussian_log_density(z, mean_mean, mean_logvar)  # [B]
    
#     # TC loss with proper scaling
#     tc_loss = (log_qz_mean.mean() - log_qz_marginal.mean()) * latent_dim / batch_size
#     tc_loss = torch.clamp(tc_loss, min=0.0)
    
#     # 2. Dimension-wise KL
#     log_pz = log_standard_normal(z)  # log p(z) under standard normal prior
#     dkl = torch.clamp(log_qz_marginal.mean() - log_pz.mean(), min=0.0)
    
#     # 3. Mutual information (index-code MI)
#     mi_loss = torch.clamp(log_qz_mean.mean() - log_qz_marginal.mean(), min=0.0)
    
#     # Properly scaled loss
#     mi_weight = 1 # 0.1  # Scale down the huge MI values you're seeing
#     vae_loss = recon_loss + beta * (mi_weight * mi_loss + tc_weight * tc_loss + dkl)
    
#     return vae_loss, recon_loss,mi_loss,tc_loss, dkl

# def gaussian_log_density(x, mean, logvar):
#     """Compute log density of Gaussian with given mean and logvar at x"""
#     return -0.5 * (np.log(2 * np.pi) + logvar + (x - mean).pow(2) / torch.exp(logvar))




#     # Sample from the posterior q(z|x)
#     z = reparameterize(mean, logvar)
    
#     # Calculate log q(z|x)
#     log_q_zx = log_normal_pdf(z, mean, logvar)
    
#     # Calculate log q(z) - marginal encoding distribution
#     # Use Monte Carlo approximation across batch
#     batch_size = mean.size(0)
#     _logqz = log_normal_pdf(z.unsqueeze(1), mean.unsqueeze(0), logvar.unsqueeze(0))
#     # Logsumexp trick for numerical stability
#     log_q_z = torch.logsumexp(_logqz.sum(dim=2), dim=1) - np.log(batch_size)
    
#     # Calculate log p(z) - prior
#     log_p_z = log_standard_normal(z)
    
#     # Calculate decomposed KL terms
#     kl_sep = log_q_zx - log_p_z  # Dimension-wise KL
#     mi_loss = (log_q_zx.mean() - log_q_z.mean())  # Mutual information
#     tc_loss = (log_q_z.mean() - log_p_z.mean())  # Total correlation
    
#     # Weighted loss
#     vae_loss = recon_loss + beta * (mi_loss + tc_weight * tc_loss + kl_sep.mean())
    
#     return vae_loss, recon_loss, mi_loss, tc_loss, kl_sep.mean()

# Helper functions
def log_normal_pdf(x, mean, logvar):
    const = torch.log(torch.tensor(2. * np.pi))
    return -.5 * (const + logvar + (x - mean).pow(2) / torch.exp(logvar))

def log_standard_normal(x):
    const = torch.log(torch.tensor(2. * np.pi))
    return -.5 * (const + x.pow(2))

def reparameterize(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std


def improved_physics_loss(energy_calculator, pos_pred, batch, mean, logvar, use_log=True):
    """
    Improved physics loss with adaptive weighting and distribution-awareness
    
    Args:
        energy_calculator: Your EnergyCalculator object
        pos_pred: Predicted positions
        batch: Batch indices
        mean, logvar: Latent distribution parameters
    """
    # Get latent space statistics
    z_var = torch.exp(logvar)
    z_var_mean = z_var.mean(dim=1, keepdim=True)
    
    # Calculate directional variance to detect collapse
    latent_cov = torch.matmul(mean.unsqueeze(2), mean.unsqueeze(1))
    directional_var = torch.diagonal(latent_cov, dim1=1, dim2=2).mean(dim=1)
    
    # Calculate standard physics components
    bond_energy, angle_energy, lj_energy = 0.0, 0.0, 0.0
    total_loss = 0.0
    num_graphs = batch.max().item() + 1
    
    for i in range(num_graphs):
        # Standard physics calculations
        mask = batch == i
        coords = pos_pred[mask]
        
        # Calculate per-graph latent statistics
        graph_z_var = z_var[i]
        graph_dir_var = directional_var[i]
        
        # Calculate energy components
        bond_e, angle_e, lj_e = energy_calculator(coords)
        
        # ---- Key Improvement: Adaptive weighting ----
        # Scale physics weight inversely with latent variance
        # This encourages exploration when the model is uncertain
        adaptive_scale = torch.sigmoid(5.0 - 10.0 * graph_z_var.mean())
        
        # Balance components dynamically
        # Bonds are most critical, then angles, then LJ
        bond_weight = 5.0 * adaptive_scale
        angle_weight = 2.0 * adaptive_scale 
        lj_weight = 0.01 * adaptive_scale
        
        # Apply weights
        physics_e = (bond_weight * bond_e + 
                     angle_weight * angle_e + 
                     lj_weight * lj_e)
        
        # Apply log transform if requested (with safe min value)
        if use_log:
            physics_e = torch.log10(physics_e + 1e-6)
            
        # Add diversity bonus for high directional variance
        # This directly counters dimensional collapse
        diversity_bonus = 0.1 * torch.sigmoid(graph_dir_var)
        physics_e = physics_e - diversity_bonus
            
        total_loss = total_loss + physics_e
        
        # Accumulate for reporting
        bond_energy += bond_e
        angle_energy += angle_e
        lj_energy += lj_e
    
    # Normalize
    if num_graphs > 0:
        total_loss = total_loss / num_graphs
        bond_energy /= num_graphs
        angle_energy /= num_graphs
        lj_energy /= num_graphs
    
    return total_loss, (bond_energy, angle_energy, lj_energy)



# def compute_tc_vae_loss(x_pred, x_true,edge_index, mean, logvar, batch, beta=0.001, tc_weight=1.0, wait = False, 
#                          mi_weight=0.1, dkl_weight=1.0, bandwidth=0.1, debug = False):
#     """
#     Complete TC-VAE loss with Monte Carlo estimation and proper scaling
    
#     Args:
#         x_pred: Reconstructed data
#         x_true: Original data
#         mean: Encoder mean output [batch_size, latent_dim]
#         logvar: Encoder log variance output [batch_size, latent_dim]
#         batch: Batch indices
#         beta: Overall KL weight
#         tc_weight: Weight for total correlation term
#         mi_weight: Weight for mutual information term
#         dkl_weight: Weight for dimension-wise KL term
#     """
#     # 1. Advanced reconstruction loss
#     recon_loss = advanced_reconstruction_loss(x_pred, x_true, edge_index, batch)

#     if debug: print(f"Reconstruction Loss: {recon_loss.item()}")
    
#     if wait:
#         return recon_loss, recon_loss,torch.tensor(0),torch.tensor(0),torch.tensor(0) # Early return if waiting for debugging


#     # Get dimensions for scaling
#     batch_size = mean.size(0)
#     latent_dim = mean.size(1)
    
#     # 2. Sample from posterior
#     z = reparameterize(mean, logvar)

#     if debug: print(f"Sampled z shape: {z.shape}, Mean shape: {mean.shape}, Logvar shape: {logvar.shape}")
    
#     # 3. Calculate log q(z|x) - log probability of z under encoder
#     log_q_zCx = log_normal_pdf(z, mean, logvar)

#     if debug: print(f"log_q_zCx shape: {log_q_zCx.shape}, Mean: {log_q_zCx.mean().item()}, Std: {log_q_zCx.std().item()}")
    
#     # 4. Calculate log p(z) - log probability of z under prior
#     log_p_z = log_standard_normal(z)

#     if debug: print(f"log_p_z shape: {log_p_z.shape}, Mean: {log_p_z.mean().item()}, Std: {log_p_z.std().item()}")

#     # 35 Calculate log q(z) using kernel density estimation
#     # This is where we apply the kernel PDE approach
#     log_q_z,joint_bandwidth = adaptive_kde(z)

#     if debug: print(f"log_q_z shape: {log_q_z.shape}, Mean: {log_q_z.mean().item()}, Std: {log_q_z.std().item()}")
    
#     # 6. Calculate log q(z_i) for each dimension i
#     log_q_z_prod = 0
#     for i in range(latent_dim):
#         log_q_z_i, _ = adaptive_kde(z[:, i:i+1])
#         if log_q_z_i.isnan().any():
#             print( "log_q_z_i IS Nan")
#             exit(1)
#         log_q_z_prod += log_q_z_i
    
#     if debug: print(f"log_q_z_prod shape: {log_q_z_prod.shape}, Mean: {log_q_z_prod.mean().item()}, Std: {log_q_z_prod.std().item()}")
#     log_q_z_prod  = torch.clamp(log_q_z_prod, max=1e3)  # Ensure positive
    
#     # # 5. Monte Carlo estimation of log q(z) - marginal log probability
#     # mat_log_qz = matrix_log_density_gaussian(z, mean, logvar)
    
#     # # Apply logsumexp for numerical stability and proper normalization
#     # # This computes log q(z) ≈ log(1/N ∑_i q(z|x_i))
#     # log_q_z = torch.logsumexp(mat_log_qz.sum(dim=2), dim=1) - np.log(batch_size)
    
#     # # 6. Calculate log q(z) assuming factorized distribution
#     # # This is sum_i log q(z_i) under marginal distributions
#     # mat_log_qz_prod = matrix_log_density_gaussian_product(z, mean, logvar)
#     # log_q_z_prod = torch.logsumexp(mat_log_qz_prod, dim=-1) - np.log(batch_size)
    
#     # 7. KL Decomposition
#     # Compute and scale terms appropriately to avoid numerical issues
    
#     # a. Index-code MI: I(z;x) = KL[q(z,x)||q(z)p(x)] = E_q(x)[KL[q(z|x)||q(z)]]
#     mi_loss = (log_q_zCx - log_q_z).mean() / latent_dim
#     mi_loss = torch.clamp(mi_loss, min=1e-8)  # Ensure positive
    
#     # b. Total correlation: TC(z) = KL[q(z)||∏_i q(z_i)]
#     tc_loss = (log_q_z - log_q_z_prod).mean() / latent_dim
#     tc_loss = torch.clamp(tc_loss, min=1e-8)  # Ensure positive
#     tc_loss = torch.clamp(tc_loss, max=1e3)  # Ensure positive

#     # c. Dimension-wise KL: ∑_i KL[q(z_i)||p(z_i)]
#     dkl_loss = (log_q_z_prod - log_p_z).mean() / latent_dim
#     dkl_loss = torch.clamp(dkl_loss, min=1e-8)  # Ensure positive
    
#     # 8. Combine with appropriate weights
#     kl_loss = mi_weight * mi_loss + tc_weight * tc_loss + dkl_weight * dkl_loss
    
#     # 9. Final loss
#     total_loss = recon_loss + beta * kl_loss
    
#     return total_loss, recon_loss, dkl_loss, tc_loss, mi_loss



# ==============================================================================
# PROPOSED NEW LOSS FUNCTIONS FOR Putils.py
#
# You should replace the existing `compute_tc_vae_loss` and its helpers
# with the functions below.
# ==============================================================================

def compute_tc_vae_loss(pos_pred, pos_true, edge_index, mean, logvar,batch,
                        beta=1.0, tc_weight=10.0, mi_weight=1.0, dkl_weight=1.0):
    """
    Computes the beta-TCVAE loss with decomposed KL terms.
    This implementation uses minibatch-weighted sampling for numerical stability,
    which is generally superior to Kernel Density Estimation for this task.

    Args:
        pos_pred (torch.Tensor): Reconstructed positions from the decoder.
        pos_true (torch.Tensor): Ground truth positions.
        edge_index (torch.Tensor): Graph connectivity.
        batch (torch.Tensor): Batch indices.
        mean (torch.Tensor): Latent space mean from the encoder.
        logvar (torch.Tensor): Latent space log-variance from the encoder.
        beta (float): The overall weight for the KL term (annealed).
        tc_weight (float): The specific weight for the Total Correlation term.
        mi_weight (float): The specific weight for the Mutual Information term.
        dkl_weight (float): The specific weight for the Dimension-wise KL term.

    Returns:
        tuple: (total_loss, recon_loss, mi_loss, tc_loss, dkl_loss)
    """
    # 1. Reconstruction Loss (using your advanced, alignment-invariant function)
    recon_loss = advanced_reconstruction_loss(pos_pred, pos_true, edge_index, batch)

    # Get dimensions
    batch_size, latent_dim = mean.shape

    # 2. Sample from posterior q(z|x) using the reparameterization trick
    z = reparameterize(mean, logvar)

    # 3. KL Divergence Decomposition
    # These calculations rely on log-probabilities of z under different distributions.
    
    # log q(z|x) for each sample
    log_q_z_given_x = log_normal_pdf(z, mean, logvar)

    # log p(z) (prior) for each sample
    log_p_z = log_standard_normal(z)
    
    # log q(z) = log [ 1/N sum_i q(z|x_i) ]
    # This is the tricky term. We estimate it using all samples in the batch.
    # We need to compute the log probability of each z_i under each q(z|x_j).
    # log_q_z_matrix has shape [batch_size (for z_i), batch_size (for x_j)]
    log_q_z_matrix = log_normal_pdf(z.unsqueeze(1), mean.unsqueeze(0), logvar.unsqueeze(0))
    
    # Log-sum-exp trick for numerical stability to compute log q(z)
    log_q_z = torch.logsumexp(log_q_z_matrix, dim=1) - math.log(batch_size)
    
    # log prod_j q(z_j) = sum_j log q(z_j)
    # This term is for the product of the marginals of q(z).
    # log_q_z_prod_matrix has shape [batch_size (for z_i), batch_size (for x_j), latent_dim]
    log_q_z_prod_matrix = log_normal_pdf(z.unsqueeze(1), mean.unsqueeze(0), logvar.unsqueeze(0), sum_dims=False)
    # Marginalize over batch dimension, then sum over latent dimension
    log_q_z_prod = torch.sum(torch.logsumexp(log_q_z_prod_matrix, dim=1) - math.log(batch_size), dim=1)

    # Decomposed KL terms (averaged over the batch)
    # a) Index-Code MI: I(z;x) = E[log q(z|x) - log q(z)]
    mi_loss = (log_q_z_given_x - log_q_z).mean()
    mi_loss = torch.clamp(mi_loss, min=1e-8)  # Ensure positive
    
    # b) Total Correlation: TC(z) = E[log q(z) - log prod_j q(z_j)]
    tc_loss = (log_q_z - log_q_z_prod).mean()
    tc_loss = torch.clamp(tc_loss, min=1e-8)  # Ensure positive

    # c) Dimension-wise KL: D_KL = E[log prod_j q(z_j) - log p(z)]
    dkl_loss = (log_q_z_prod - log_p_z).mean()
    dkl_loss = torch.clamp(dkl_loss, min=1e-8)  # Ensure positive
    # 4. Combine into final loss
    kl_loss = mi_weight * mi_loss + tc_weight * tc_loss + dkl_weight * dkl_loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, dkl_loss, tc_loss, mi_loss

# --- Helper functions for the loss calculation ---

def reparameterize(mean, logvar):
    """Standard reparameterization trick."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std

def log_normal_pdf(x, mean, logvar, sum_dims=True):
    """
    Calculates log probability of x under a normal distribution.
    Args:
        x, mean, logvar: Tensors of same shape.
        sum_dims (bool): If True, sums the log probabilities over the last dimension.
    """
    const = -0.5 * math.log(2 * math.pi)
    log_prob = const - 0.5 * logvar - 0.5 * ((x - mean)**2 / torch.exp(logvar))
    if sum_dims:
        return torch.sum(log_prob, dim=-1)
    return log_prob

def log_standard_normal(x):
    """Calculates log probability of x under a standard normal distribution."""
    return log_normal_pdf(x, torch.zeros_like(x), torch.zeros_like(x))


# ==============================================================================
# PROPOSED NEW PHYSICS LOSS FUNCTION FOR Putils.py
# ==============================================================================

def compute_physics_loss(energy_calculator, pos_pred, batch, 
                         bond_weight=1.0, angle_weight=0.5, lj_weight=0.1, use_log=False):
    """
    Computes a physics-based loss using an energy calculator.

    Args:
        energy_calculator: Instantiated object that can compute energies.
        pos_pred (torch.Tensor): Predicted positions from the decoder.
        batch (torch.Tensor): Batch indices.
        bond_weight (float): Weight for the bond energy term.
        angle_weight (float): Weight for the angle energy term.
        lj_weight (float): Weight for the Lennard-Jones (non-bonded) term.
        use_log (bool): If True, applies a log transform to the energies to
                        prevent extreme gradients from high-energy structures.

    Returns:
        tuple: (total_physics_loss, avg_bond_e, avg_angle_e, avg_lj_e)
    """
    total_loss = 0.0
    total_bond_e, total_angle_e, total_lj_e = 0.0, 0.0, 0.0
    num_graphs = batch.max().item() + 1

    for i in range(num_graphs):
        mask = (batch == i)
        coords = pos_pred[mask] # Get coords for the i-th graph

        # Assumes your energy_calculator returns a tuple of energies
        bond_e, angle_e, lj_e = energy_calculator(coords)

        # Log transform is crucial for stability! It punishes high energies
        # without creating exploding gradients.
        if use_log:
            bond_e = torch.log1p(bond_e)
            angle_e = torch.log1p(angle_e)
            lj_e = torch.log1p(lj_e)
        
        graph_loss = (bond_weight * bond_e +
                      angle_weight * angle_e +
                      lj_weight * lj_e)
        
        total_loss += graph_loss
        total_bond_e += bond_e
        total_angle_e += angle_e
        total_lj_e += lj_e

    # Average over the batch
    if num_graphs > 0:
        total_loss /= num_graphs
        total_bond_e /= num_graphs
        total_angle_e /= num_graphs
        total_lj_e /= num_graphs
    
    return total_loss, total_bond_e, total_angle_e, total_lj_e