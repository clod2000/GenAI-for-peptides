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

from create_full_graph_data import TrajectoryDataset

import sys


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
                return_max_position = False
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
    Returns:
        dataset (InMemoryDataset): The processed dataset containing the atomic positions and features.
    """ 

    
    if verbose: print("Loading dataset ...")
    if verbose: print()
    
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

        features = torch.cat((one_hot_encoded,features[:,1:]),dim=1)
    
    else:
        if verbose: print("Not including atom features, discarding it ...")
        features = features[:,1:]

    if scale_features:

        if verbose: print(f"Scaling features ...")

        # Scale the features to have zero mean and unit variance
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features.numpy())
        features = torch.tensor(scaled_features, dtype=torch.float32)


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

