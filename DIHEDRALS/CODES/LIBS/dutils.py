import torch
import os
import torch._numpy as np
import pandas as pd

from torch_geometric.data import Data
from torch.utils.data import random_split

from torch_geometric.data import Data, InMemoryDataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch_geometric.transforms as T
import torch.nn.functional as F

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


##### Dataset creation and transformation functions #####
def create_dataset(file_csv, verbose=False):
    """
    Create a dataset from a CSV file containing dihedral angles.
    
    Args:
        file_csv (str): Path to the CSV file.
        verbose (bool): If True, print additional information.
    
    Returns:
        torch.Tensor: Tensor containing the dihedral angles.
    """
        
    if verbose: print(f"Reading dihedral angles from {file_csv}")
    try: df = pd.read_csv(file_csv)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_csv} not found.")
    
    if verbose: print(f"Data shape: {df.shape}")

    # Reorder the columns as [Phi_1, Psi_1, Omega_1, Phi_2, Psi_2, Omega_2, Phi_3, Psi_3, Omega_3, Phi_4, Psi_4] 
    df = df[['Phi_1', 'Psi_1', 'Omega_1', 'Phi_2', 'Psi_2', 'Omega_2', 'Phi_3', 'Psi_3', 'Omega_3', 'Phi_4', 'Psi_4']]

    if verbose: print(f"Reordered data shape: {df.shape}")
    if verbose: print(f"Data columns: {df.columns.tolist()}")
    if verbose: print(f"Dataframe head: {df.head()}")

    edge_index = torch.tensor([[i for i in range(0, len(df.columns)-1)], [j for j in range(1,len(df.columns))]], dtype=torch.long)
    if verbose: print(f"Edge index shape: {edge_index.shape}")

    x_deg = torch.tensor(df.values, dtype=torch.float)
    if verbose: print(f"degree angles shape: {x_deg.shape}")
    x_rad = torch.deg2rad(x_deg)
    x_sin = torch.sin(x_rad).unsqueeze(2)
    x_cos = torch.cos(x_rad).unsqueeze(2)
    x = torch.cat([x_sin, x_cos], dim=2)

    if verbose: print(f"Converted angles to radians, sin and cos shapes: {x_sin.shape}, {x_cos.shape}, combined shape: {x.shape}")


    data_list = [Data(x = xi ,edge_index = edge_index) for xi in x]

    dataset = InMemoryDataset(root='../DATA/', transform=None)
    dataset.data, dataset.slices = dataset.collate(data_list)

    if verbose: print(f"Dataset created with {len(dataset)} samples, {dataset.x.shape[1]} features and {dataset.edge_index.shape[1]} edges.")


    return dataset

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


def cyclic_annealing(epoch, start_value=0.1, end_value=1.0, cycle_length=10):
    """
    Cyclic annealing function that oscillates between start_value and end_value.
    
    Args:
        epoch (int): Current epoch number.
        start_value (float): Starting value of the annealing.
        end_value (float): Ending value of the annealing.
        cycle_length (int): Length of one cycle in epochs.
        
    Returns:
        float: Annealed value for the current epoch.
    """
    # cycle = (epoch // cycle_length) % 2
    # if cycle == 0:
    return start_value + (end_value - start_value) * (epoch % cycle_length) / cycle_length
    # else:
    #    return end_value - (end_value - start_value) * (epoch % cycle_length) / cycle_length



# useful 

def kl_loss(mu, logstd):
    """
    Compute the KL divergence loss.
    """
    kl = -0.5 * torch.mean(1 + logstd - mu.pow(2) - logstd.exp())
    return kl  # Normalized by batch size

def angle_loss(pred, target, batch = None):  # use sin and cos to compute the angle loss to avoid discontinuity   

    """ Compute the angle loss between predicted and target angles.
    Args:
        pred (torch.Tensor): Predicted angles in radians, shape [batch_size, num_nodes, 2].
        target (torch.Tensor): Target angles in radians, shape [batch_size, num_nodes, 2].  
        batch (torch.Tensor, optional): Batch indices for each node, shape [batch_size]. If provided, reshapes pred and target.
        Assuming pred and target are in sin cos form
    """
    if batch is not None:
        # reshape to [batch_size, num_nodes, 2] if batch is provided
        pred = pred.view(batch.size(0), -1, 2)
        target = target.view(batch.size(0), -1, 2)
        

        return F.mse_loss(pred, target)

def compute_loss(model, x_recon, x, beta=1.0, learn_beta=False):
    
    angles_loss = angle_loss(x_recon, x)
    kl_loss = model.kl_loss()  # KL divergence loss for regularization
    if learn_beta:
        raise NotImplementedError("Learning beta is not implemented yet.")
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


############## Analysis and Visualization Functions ##############

def ramachandran_plot(df_pred, df_true, color = 'blue', save_path=None):
    """
    Create a Ramachandran plot for each residue in the dataset.
    Each residue is represented by a scatter plot of its phi and psi angles.    
    Parameters:
    df_pred (pd.DataFrame): DataFrame containing predicted phi and psi angles for residues.
    df_true (pd.DataFrame): DataFrame containing true phi and psi angles for residues.
    """

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    for i in range(4):
        axs[i].scatter(df_true[f'Phi_{i+1}'], df_true[f'Psi_{i+1}'], alpha=0.2, color='orange', label='True', s=10)
        axs[i].scatter(df_pred[f'Phi_{i+1}'], df_pred[f'Psi_{i+1}'], alpha=0.7, color=color, label='Predicted', s=10)

        axs[i].set_title(f'Residue {i+1}')
        axs[i].set_xlabel('Phi')
        axs[i].set_ylabel('Psi')
        axs[i].set_xlim(-180, 180)
        axs[i].set_ylim(-180, 180)
        axs[i].grid(True)
        axs[i].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:   
        plt.show()

def test_generation_ramachandran(model,file_csv, num_samples=1000, save_path=None ):
    """    Generate samples from the decoder and plot the Ramachandran plot.
    Args:
        model (DVGAE): The trained DVGAE model. 
        file_csv (str): Path to the CSV file containing the dihedral angles ( to be used as ground truth).
        num_samples (int): Number of samples to generate for the Ramachandran plot.
    """
    gen_data = []
    for _ in range(num_samples):
        model.eval()
        z = torch.randn(1, model.latent_dim).to(model.device)  # random latent vector for testing the decoder
        x_recon = model.decoder(z)  # get the reconstructed data from the decoder
        gen_data.append(inverse_transform(x_recon).detach().cpu().squeeze().numpy()) 

    columns = ['Phi_1', 'Psi_1', 'Omega_1', 'Phi_2', 'Psi_2', 'Omega_2', 'Phi_3', 'Psi_3', 'Omega_3', 'Phi_4', 'Psi_4']
    df_gen = pd.DataFrame(gen_data, columns=columns)
    df_data = pd.read_csv(file_csv)
    df_data = df_data[columns]

    ramachandran_plot(df_gen, df_data, color='green', save_path=save_path)

def test_reconstruction_ramachandran(model, test_set, file_csv, save_path=None):
    """
    Test the reconstruction of the model on a test set and plot the Ramachandran plot.        
    Args:
        model (DVGAE): The trained DVGAE model.
        test_set (InMemoryDataset): The dataset to use for testing.
        num_samples (int): Number of samples to generate for the Ramachandran plot.
        save_path (str, optional): Path to save the plot. If None, the plot will be shown.
    """
    model.eval()
    recon_data = []
    with torch.no_grad():
        for data in test_set:
            data = data.to(model.device)
            x_recon,_,_ = model(data.x, data.edge_index, data.batch)  # get the reconstructed data from the model
            
            recon_data.append(inverse_transform(x_recon).detach().cpu().squeeze().numpy())

    columns = ['Phi_1', 'Psi_1', 'Omega_1', 'Phi_2', 'Psi_2', 'Omega_2', 'Phi_3', 'Psi_3', 'Omega_3', 'Phi_4', 'Psi_4']
    df_recon = pd.DataFrame(recon_data, columns=columns)
    df_data = pd.read_csv(file_csv)
    df_data = df_data[columns]
    ramachandran_plot(df_recon, df_data, save_path=save_path)



def analyze_latent_space(model, dataloader, device, file_path, verbose=True):
    """Encodes the dataset, performs PCA, and plots the latent space colored by energy."""
    model.eval()
    latent_vectors = []
  
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Analyzing Latent Space", leave=False):
            data = data.to(device)
            mean, _ = model.encoder(data.x, data.edge_index, data.batch)  # get the mean from the encoder
            latent_vectors.append(mean.cpu())

    latent_vectors = torch.cat(latent_vectors, dim=0).numpy()
    

    # Use PCA to reduce to 2D if latent dimension is greater than 2
    if latent_vectors.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        latent_vectors = pca.fit_transform(latent_vectors)
    

    plt.figure(figsize=(10, 8))
    points = plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], s=20, alpha=0.7)
    plt.title('Latent Space Visualization of Test Set (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"{file_path}/latent_space_pca.png")
    plt.close()


import math

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
    #recon_loss = recon_loss(pos_pred, pos_true, edge_index, batch)

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
    #total_kl_loss = mi_weight * mi_loss + tc_weight * tc_loss + dkl_weight * dkl_loss
    #total_loss = recon_loss + beta * kl_loss

    return dkl_loss, tc_loss, mi_loss

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

