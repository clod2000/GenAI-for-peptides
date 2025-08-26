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
import sys
import argparse 
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

sys.path.append("LIBS")
from LIBS.utils import *
from LIBS.Putils import create_physics_informed_dataset
from LIBS.new_PIGVAE import *
from LIBS.PIGVAE_off import *

print ("Importing force field module...")
from LIBS.force_field import *
#from LIBS.upgraded_ff import *
print("Force field module imported successfully.")


# Create a single parser with both arguments
parser = argparse.ArgumentParser(description='Full Graph VAE with EGNN')
parser.add_argument('--config', type=str, default='config.template.in', help='Path to the configuration file')
parser.add_argument('--verbose', action='store_true', default=True, help='Enable verbose mode')

# Parse the command line arguments
args = parser.parse_args()
config_file = args.config
verbose = args.verbose

#verbose= True  # set to debug for now, will be set to False in the future

#Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if verbose: print(f"Using device: {device}")
if verbose: print(f"Using config file: {config_file}")

# Read the parameters from the config file
try:
    config = parse_config(config_file=config_file, verbose=verbose)
    if verbose: print(f"Configuration file {config_file} read successfully.")
except Exception as e:
    print(f"Error reading config file: {e}")
    sys.exit(1)

if verbose: print(f"Configuration parameters: {config}")

debug = False
weight_mean_debug = False # if True, the mean weight gradient is printed during training, used for debugging purposes


# EXTRA PARAMETERS NOT IN THE CONFIG FILE
OFFICIAL_EGNN = False
AE = False # if True, the model is trained as an autoencoder, otherwise it is trained as a VAE
FREEZE_EXCEPT_LOGVAR = False # if True , all the parameters of the model are frozen except the log variance, used to switch from AE to VAE trainings

# Architecture parameters
MODEL_ARCHITECTURE = config.get('MODEL_ARCHITECTURE', 'original') # architecture of the model, can be 'original' or 'hybrid_displacement'
ENCODER_POS_PROJECTION_DIM = config.get('ENCODER_POS_PROJECTION_DIM', 64) # dimension of the position projection in the encoder, used to project the positions to a lower dimension if 'hybrid_displacement' is used

# Training parameters
TRAINING_MODE = config.get('TRAINING_MODE', 'generative') # 'denoising' or 'generative'
NOISE_LEVEL = config.get('NOISE_LEVEL', 0.1) # e.g., 0.1 nm noise
NOISE_ANNEALING = config.get('NOISE_ANNEALING', True) # if True, the noise level is annealed during training, otherwise it is fixed

#### model parameters
# encoder
ENCODER_TYPE = config.get('ENCODER_TYPE', 'standard') # type of the encoder, can be 'standard' or 'denoise' # it acts on the pos 
NOISE_LV = config.get('NOISE_LV', 0.1) # noise level for the denoising encoder, used to create a noisy version of the positions
HIDDEN_ENCODER_CHANNELS = config.get('HIDDEN_ENCODER_CHANNELS', 256)
OUT_ENCODER_CHANNELS = config.get('OUT_ENCODER_CHANNELS', 128)
NUM_ENC_LAYERS = config.get('NUM_ENC_LAYERS', 5) # number of EGNN layers in the encoder
ATTENTION_ENCODER = config.get('ATTENTION_ENCODER', True) # if True, attention is used in the encoder
LATENT_DIM = config.get('LATENT_DIM', 128) # latent dimension of the encoder, used to create the latent space
TANH_ENCODER = config.get('TANH_ENCODER', True) # if True, the output of the encoder is passed through a tanh activation function (for positions)
NORMALIZE_ENCODER = config.get('NORMALIZE_ENCODER', True) # if True, the encoder output is normalized
# decoder
MLP_DECODER_POS_SIZE = config.get('MLP_DECODER_POS_SIZE', [256,256,128]) # size of the MLP decoder for positions
HIDDEN_DECODER_CHANNELS = config.get('HIDDEN_DECODER_CHANNELS', 256)
NUM_DEC_LAYERS = config.get('NUM_DEC_LAYERS', 5)
ATTENTION_DECODER = config.get('ATTENTION_DECODER', True)
TANH_DECODER = config.get('TANH_DECODER', True) # if True, the output of the decoder is passed through a tanh activation function (for positions)
NORMALIZE_DECODER = config.get('NORMALIZE_DECODER', True) # if True, the decoder output is normalized

#### dataset parameters
INCLUDE_ATOM_TYPE = config.get('INCLUDE_ATOM_TYPE', True) # if True, the atom type is included in the dataset
SCALE_FEATURES = config.get('SCALE_FEATURES', True)
SCALE_POSITIONS = config.get('SCALE_POSITIONS', True) # if True, the positions are scaled to the range [0, 1]
INITIAL_ALIGNMENT = config.get('INITIAL_ALIGNMENT', True) # if True, the dataset is aligned to the initial positions

#### training parameters
EPOCHS = config.get('EPOCHS', 50)
BATCHSIZE = config.get('BATCHSIZE', 64)
LEARNING_RATE = config.get('LEARNING_RATE', 1E-4)
WARMUP_EPOCHS = config.get('WARMUP_EPOCHS', 0)
WEIGHT_DECAY = config.get('WEIGHT_DECAY', 0) # weight decay for the optimizer, set to 0 to disable weight decay ( bad idea using it for vae)
return_pos_angstrom = config.get('return_pos_angstrom', False) # if True, the positions are returned in Angstrom, otherwise they are returned in the range [0, 1]


advanced_recon_loss = config.get('advanced_recon_loss', True) # if True, the advanced reconstruction loss is used, otherwise the standard reconstruction loss is used
advanced_kl_loss = config.get('advanced_kl_loss', True) # if True, the advanced KL divergence loss is used, otherwise the standard KL divergence loss is used

tc_weight = config.get('tc_weight', 10.0) # weight for the total correlation loss in the advanced KL divergence loss
mi_weight = config.get('mi_weight', 1.0) # weight for the mutual information loss in the advanced KL divergence loss
dkl_weight = config.get('dkl_weight', 1.0) # weight for the dimension-wise KL divergence loss in the advanced KL divergence loss

#### Scheduler parameters
USE_SCHEDULER = config.get('USE_SCHEDULER', False) # if True, the learning rate scheduler is used
SCHEDULER_PATIENCE = config.get('SCHEDULER_PATIENCE', 10) # number of epochs with no improvement after which learning rate will be reduced
SCHEDULER_FACTOR = config.get('SCHEDULER_FACTOR', 0.5) # factor by which the learning rate will be reduced. new_lr = lr * factor
SCHEDULER_TYPE = config.get('SCHEDULER_TYPE', 'ReduceLROnPlateau') # type of the scheduler, can be 'CosineAnnealingLR' or 'StepLR' or 'ReduceLROnPlateau'
SCHEDULER_THRESHOLD = config.get('SCHEDULER_THRESHOLD', 0.0001) # threshold for the scheduler, used to stop the training if the loss is below this value

# Beta annealing parameters
BETA = config.get('BETA', None)
CYCLIC_BETA = config.get('CYCLIC_BETA', False) # if True, the beta is cycled between beta_min and beta_max
CYCLE_LENGTH = config.get('CYCLE_LENGTH', 10) # length of the cycle for cyclic beta annealing
wait_epochs = config.get('wait_epochs', 0)
annealing_epochs = config.get('annealing_epochs', 50)
beta_min = config.get('beta_min', 0.00001)
beta_max = config.get('beta_max', 0.0001)

# force field parameters
USE_FORCE_FIELD = config.get('USE_FORCE_FIELD', True) # if True, the force field is used to calculate the energy of the system
PDB_FOR_ENERGY = config.get('PDB_FOR_ENERGY', '../DATA/raw/protein_only.pdb') # path to the PDB file for energy calculation
LAMBDA_ENERGY = config.get('LAMBDA_ENERGY', None) # weight for the energy loss in the total loss function
wait_lambda_epochs = config.get('wait_lambda_epochs', 10) # number of epochs to wait before starting to use the force field in the loss function
lambda_annealing_epochs = config.get('lambda_annealing_epochs', 50) # number of epochs to anneal the lambda parameter
lambda_min = config.get('lambda_min', 1e-15) # minimum value for the lambda parameter
lambda_max = config.get('lambda_max', 0.001) # maximum value for the lambda parameter

USE_LOG_FF = config.get('USE_LOG_FF', True) # if True, the force field loss is calculated using log scaling, otherwise it is calculated using the raw values
USE_BOND_FF = config.get('USE_BOND_FF', True) # if True, the bonded energy is included in the force field loss
USE_ANGLE_FF = config.get('USE_ANGLE_FF', True) # if True, the angle energy is included in the force field loss
USE_LJ_FF = config.get('USE_LJ_FF', True) # if True, the Lennard-Jones energy is included in the force field loss


# Other parameters
DISABLE_TQDM = config.get('DISABLE_TQDM', False) # if True, the tqdm progress bar is disabled
SEED = config.get('SEED', 42) # seed for reproducibility
NAME_SIMULATION = config.get('NAME_SIMULATION', None ) # name of the simulation, used to create a folder to save the model
NAME_FOLDER = config.get('NAME_FOLDER', 'template') # name of the folder to save the model, if None, the folder will be created with the name of the model architecture and encoder type
CONTINUE_FROM = config.get('CONTINUE_FROM', None) # set the path to the model to continue training from, if None, the training will start from scratch
STARTING_EPOCH = config.get('STARTING_EPOCH', 0) # if CONTINUE_FROM is not None, the training will start from this epoch
ALIGN_RECONS_LOSS = config.get('ALIGN_RECONS_LOSS', True) # if True, samples are aligned before computing the reconstruction loss, otherwise the reconstruction loss is computed without alignment
TEST_MODEL = config.get('TEST_MODEL', False) # if True, the model is tested after training
MIN_KL = config.get('MIN_KL', 0.0001) # minimum value for the KL divergence loss, if the KL divergence is below this value, the total loss is set to the reconstruction loss only


if CONTINUE_FROM is None: 
    STARTING_EPOCH = 0 # if CONTINUE_FROM is None, the training will start from epoch 0
    if verbose: print("Starting training from scratch")
else:
    if verbose: print(f"Continuing training from {CONTINUE_FROM} at epoch {STARTING_EPOCH}")
    if not os.path.exists(CONTINUE_FROM):
        raise FileNotFoundError(f"The path {CONTINUE_FROM} does not exist. Please check the path and try again.")
    

# Set the random seed for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)


#device = torch.device( 'cpu')

# Create dataset and dataloaders

if USE_FORCE_FIELD:
    return_pos_angstrom = False  # if using force field, return positions in nm for energy calculation
    print(f"Using force field, returning positions in Angstrom: {return_pos_angstrom}")

max_positions = None # Initialize max_positions
if SCALE_POSITIONS:
    # In this case I need to resize the dataset to true positions to compute the force field and the energy
    dataset, max_positions = get_dataset(
    include_atom_type=INCLUDE_ATOM_TYPE,
    scale_features=SCALE_FEATURES,
    scale_pos=SCALE_POSITIONS,
    initial_alignment=INITIAL_ALIGNMENT,
    verbose=verbose,
    return_max_position=True,
    return_pos_angstrom=return_pos_angstrom
    )
    max_positions = max_positions.to(device)  # Move max_positions to the device
else:
    dataset = get_dataset(
        include_atom_type=INCLUDE_ATOM_TYPE,
        scale_features=SCALE_FEATURES,
        scale_pos=SCALE_POSITIONS,
        initial_alignment=INITIAL_ALIGNMENT,
        verbose=verbose,
        return_max_position=False,
        return_pos_angstrom=return_pos_angstrom
    )
    # Define max_positions for consistency in function calls, even if not used for scaling
    max_positions = torch.ones(3, device=device)

# Initialize the force field if needed
physics_critic = None
if USE_FORCE_FIELD:
    if verbose: print("Using force field for energy calculation")
    if verbose: print(f"Using PDB file for energy calculation: {PDB_FOR_ENERGY}")
    # Initialize the energy calculator with the PDB file and force field files
    physics_critic = EnergyCalculator(
        pdb_file=PDB_FOR_ENERGY
    )

    #dataset = add_physics_attributes_to_dataset(dataset, physics_critic)
    dataset = create_physics_informed_dataset(dataset, physics_critic)
    if verbose: print(f"Dataset with physics attributes created with {len(dataset)} graphs")
    
# Calculate the mean structure if needed
pos_ref = None
if MODEL_ARCHITECTURE == 'hybrid_displacement':
    if verbose: print("Calculating mean reference structure for the hybrid model...")

    Aligned_dataset = get_dataset(include_atom_type=INCLUDE_ATOM_TYPE,
                             scale_features=SCALE_FEATURES,
                             scale_pos=SCALE_POSITIONS,
                             initial_alignment=True,
                             return_pos_angstrom=False) 



    all_pos = torch.stack([data.pos for data in Aligned_dataset], dim=0)  # Shape: (num_graphs, num_atoms, 3)
    pos_ref = all_pos.mean(dim=0).to(device)  # Shape: (num_atoms, 3)

    
    if verbose: print(f"Reference structure created with shape: {pos_ref.shape}")
    if verbose: print(f"Single graph reference shape: {pos_ref.shape}")
    if verbose: print(f"Used {len(dataset)} graphs to calculate mean structure")
    if verbose: print()


if OFFICIAL_EGNN:
    if verbose: print("Using official EGNN implementation")
    model = FGVAE(
        encoder=Official_EGNN_Encoder(
            in_channels=dataset[0].num_features,
            hidden_channels=HIDDEN_ENCODER_CHANNELS,
            num_egnn_layers=NUM_ENC_LAYERS,
            latent_dim=LATENT_DIM,
            mode= ENCODER_TYPE,
            attention = ATTENTION_ENCODER,
            debug= debug
        ),
        decoder=Official_EGNN_Decoder(
            latent_dim=LATENT_DIM,
            node_feature_dim_initial=dataset[0].num_features,
            hidden_channels=HIDDEN_DECODER_CHANNELS,
            num_egnn_layers=NUM_DEC_LAYERS,
            architecture= MODEL_ARCHITECTURE,
            attention = ATTENTION_DECODER,
            debug= debug
            
        ),
        AE = AE
    ).to(device)
else:
    if verbose: print("Using custom EGNN implementation (from github)")

    # Create the model
    model = FGVAE(
            encoder=EGNN_Encoder(
                in_channels=dataset[0].num_features,
                hidden_channels_egnn=HIDDEN_ENCODER_CHANNELS,
                out_channels_egnn=OUT_ENCODER_CHANNELS,
                num_egnn_layers=NUM_ENC_LAYERS,
                latent_dim=LATENT_DIM,
                attention=ATTENTION_ENCODER,
                architecture=MODEL_ARCHITECTURE,
                mode=ENCODER_TYPE,  
                pos_projection_dim=ENCODER_POS_PROJECTION_DIM,
                tanh=TANH_ENCODER,
                normalize=NORMALIZE_ENCODER,
                verbose=verbose,
                edge_dim= dataset[0].edge_attr.size(1) if dataset[0].edge_attr is not None else 0
            ),
            decoder=EGNN_Decoder(
                latent_dim=LATENT_DIM,
                node_feature_dim_initial=dataset[0].num_features,
                hidden_nf=HIDDEN_DECODER_CHANNELS,
                num_egnn_layers=NUM_DEC_LAYERS,
                attention=ATTENTION_DECODER,
                architecture=MODEL_ARCHITECTURE,
                pos_MLP_size=MLP_DECODER_POS_SIZE,
                tanh=TANH_DECODER,
                normalize=NORMALIZE_DECODER,
                edge_dim= dataset[0].edge_attr.size(1) if dataset[0].edge_attr is not None else 0,
                verbose=verbose
            ),
            AE = AE
        ).to(device)

if verbose: print_model_summary(model)

if CONTINUE_FROM is not None:
    print(f"Loading model from {CONTINUE_FROM}")
    if not os.path.exists(CONTINUE_FROM):
        raise FileNotFoundError(f"The path {CONTINUE_FROM} does not exist. Please check the path and try again.")
    model.load_state_dict(torch.load(CONTINUE_FROM, map_location=device))
    print("Model loaded successfully.")



if FREEZE_EXCEPT_LOGVAR:
    # --- FREEZING LOGIC ---
    print("Freezing all layers except encoder.fc_log_var...")
    for name, param in model.named_parameters():
        param.requires_grad = False # Freeze everything by default

    for name, param in model.encoder.to_latent_logvar.named_parameters():
        param.requires_grad = True # Un-freeze only this layer's parameters
        print(f"  - Unfrozen: {name}")

    for name, param in model.encoder.to_latent_mean.named_parameters():
        param.requires_grad = True
        print(f"  - Unfrozen: {name}")

    # Create a new optimizer that only sees the unfrozen parameters
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=LEARNING_RATE # Use a standard LR for this simple task
    )

else:
# Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


# Create the scheduler
if USE_SCHEDULER:
    if SCHEDULER_TYPE == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS) # never used, could need  other implementation
    elif SCHEDULER_TYPE == 'StepLR':
        scheduler = StepLR(optimizer, step_size=SCHEDULER_PATIENCE, gamma=SCHEDULER_FACTOR)
    elif SCHEDULER_TYPE == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, threshold=SCHEDULER_THRESHOLD, min_lr=1e-8)
    else: 
        raise ValueError(f"Unknown scheduler type: {SCHEDULER_TYPE}")

# Define the path to save the models, use numbers to define new simulations 
if NAME_SIMULATION is not None:
    file_path = f'../RUNS/{MODEL_ARCHITECTURE}/{NAME_FOLDER}/{NAME_SIMULATION}/'
    if os.path.exists(file_path):
        print(f"Directory {file_path} already exists. Adding # to the name.")
        # If the directory already exists, append a number to the name
        i = 0
        while os.path.exists(file_path):
            i += 1
            file_path = f'../RUNS/{MODEL_ARCHITECTURE}/{NAME_FOLDER}/{NAME_SIMULATION}_{i}/'
        print(f"Creating directory {file_path}")
        os.makedirs(file_path)
else:
    i = 1
    while True:
        if not os.path.exists(f'../RUNS/{MODEL_ARCHITECTURE}/{NAME_FOLDER}/simulation_{i}'):
            print(f"Creating directory for simulation {i}")
            os.makedirs(f'../RUNS/{MODEL_ARCHITECTURE}/{NAME_FOLDER}/simulation_{i}')
            file_path = f'../RUNS/{MODEL_ARCHITECTURE}/{NAME_FOLDER}/simulation_{i}/'
            break
        i += 1

if not os.path.exists(file_path):
    print(f"Creating directory {file_path}")
    os.makedirs(file_path)
# Copy the config file to the model folder
os.system(f'cp {config_file} {file_path}')

# Create a SummaryWriter to log the training process
writer = SummaryWriter(log_dir=file_path)   

# Training loop

lr = LEARNING_RATE

if CONTINUE_FROM is None:
    train_force_loss = torch.inf
else:
    train_force_loss = 0.0

if MODEL_ARCHITECTURE == 'hybrid_displacement':
    print(f"\nSTARTING TRAINING IN {TRAINING_MODE} MODE...\n")

if MODEL_ARCHITECTURE == 'original':
    print(f"\nSTARTING TRAINING with {ENCODER_TYPE} encoder ...\n")



train_loader, val_loader, test_loader = get_dataloaders(
    dataset=dataset,
    shuffle=True,
    seed=SEED,
    batch_size=BATCHSIZE,
    verbose=verbose
)

   

for epoch in range(STARTING_EPOCH, STARTING_EPOCH + EPOCHS):
    # random pos_ref from the dataset
    #pos_ref = train_loader.dataset[np.random.randint(len(train_loader.dataset))].pos.to(device)new_egnn.sh

  

    # --- LR WARM-UP LOGIC ---
    if WARMUP_EPOCHS > 0 and epoch < WARMUP_EPOCHS:
        # Linearly increase the learning rate
        lr_scale = (epoch + 1) / WARMUP_EPOCHS
        for param_group in optimizer.param_groups:
            param_group['lr'] = LEARNING_RATE * lr_scale
    # After warm-up, the LR is the standard one (and can be controlled by a scheduler)
    

    if USE_SCHEDULER and epoch >= WARMUP_EPOCHS:
        if lr > scheduler.get_last_lr()[0]:
            print(f"Adjusting learning rate from {lr} to {scheduler.get_last_lr()[0]}")
            lr = scheduler.get_last_lr()[0]

    # annealers

    if BETA is not None:
        beta = BETA
    elif not CYCLIC_BETA:
        beta = beta_annealer(epoch,beta_min, beta_max, annealing_epochs,wait_epochs )
    else:
        beta = cyclic_annealing(epoch, beta_min, beta_max, CYCLE_LENGTH)
        print(f"Using cyclic beta annealing: {beta}")

    if USE_FORCE_FIELD:

        if epoch < wait_lambda_epochs:
            lambda_energy = 0.0
        elif LAMBDA_ENERGY is not None:
            lambda_energy = LAMBDA_ENERGY
        else:
            lambda_energy = beta_annealer(epoch, lambda_min, lambda_max, lambda_annealing_epochs, wait_lambda_epochs)


    ##### to be set in a more clean way
    if train_force_loss is torch.inf and LAMBDA_ENERGY is None:
        lambda_energy = 0.0
    else:
      lambda_energy = LAMBDA_ENERGY if LAMBDA_ENERGY is not None else lambda_energy

       
    train_pbar = tqdm(train_loader, disable=DISABLE_TQDM, desc=f"Epoch {epoch+1}/{STARTING_EPOCH+EPOCHS} [Train]", leave=False)

    # Training
    model.train()
    train_loss = 0
    train_kl_loss = 0
    train_kl_sep_loss = 0
    train_recon_loss = 0
    train_mi_loss = 0
    train_tc_loss = 0

    if USE_FORCE_FIELD:
        train_force_loss = 0

    mean_train = []
    log_var_train = []

  

    for data in train_pbar:
        data = data.to(device)
         

        if TRAINING_MODE == 'denoising': # Not used anymore
            # Create a noisy version of the BATCH's target positions
            pos_ref_for_decoder = data.pos + torch.randn_like(data.pos) * NOISE_LEVEL
        elif TRAINING_MODE == 'generative' and pos_ref is not None:
            # Use the single, averaged reference for the whole batch
            num_graphs_in_batch = data.batch.max().item() + 1
            pos_ref_for_decoder = pos_ref.repeat(num_graphs_in_batch, 1)
        elif TRAINING_MODE == 'generative' and pos_ref is None:
            pos_ref_for_decoder = None
        else:   
            raise ValueError(f"Unknown TRAINING_MODE: {TRAINING_MODE}")


        optimizer.zero_grad()

        noise = NOISE_LEVEL if not NOISE_ANNEALING else denoise_annealing(epoch, EPOCHS, start_value= NOISE_LEVEL, end_value=0.0)

        pos_pred, mean, log_var, batch_vec = model(data, pos_ref=pos_ref_for_decoder, new_noise_level=noise)

        # # Basic losses
        if not advanced_kl_loss and not AE:
            kl_loss = KL_divergence(mean, log_var)
            if advanced_recon_loss:
                recon_loss = advanced_reconstruction_loss(pos_pred, data.pos, data.edge_index, data.batch, align_coords=ALIGN_RECONS_LOSS) 
            else:
                recon_loss = reconstruction_loss(pos_pred, data.pos, data.batch, align=ALIGN_RECONS_LOSS)

            if kl_loss < MIN_KL: # free-bits KL loss
                total_loss = recon_loss
            else:
                total_loss = recon_loss + beta * kl_loss

        elif not advanced_kl_loss and AE:
            # In this case, we compute the AE loss, which is just the reconstruction loss
            recon_loss = reconstruction_loss(pos_pred, data.pos, data.edge_index, data.batch, align=ALIGN_RECONS_LOSS)
            total_loss = recon_loss
            kl_loss = torch.tensor(0.0)

        elif advanced_kl_loss:
            # Advanced losses
            # Here we compute the advanced TC-VAE loss, which includes the mutual information and the total correlation
            # This is done by computing the TC-VAE loss with the mean and log variance of the encoder
            # The TC-VAE loss is computed as follows:
            # 1. Compute the reconstruction loss
            # 2. Compute the KL divergence between the mean and log variance
            # 3. Compute the mutual information and total correlation
            # 4. Compute the total loss as the sum of the reconstruction loss, KL divergence, mutual information and total correlation

            # Here with kl_loss we mean the total loss, which includes the mutual information and total correlation, the name is just for modularity with the simple KL divergence loss

            wait = epoch < wait_epochs  # If we are waiting for the beta annealing to start, set wait to True
            if AE:
                total_loss, recon_loss, kl_sep, tc_loss, mi_loss = torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0) 
                total_loss = advanced_reconstruction_loss(pos_pred, data.pos, data.edge_index, data.batch, align_coords=ALIGN_RECONS_LOSS)
                total_loss = recon_loss
            
            elif FREEZE_EXCEPT_LOGVAR:
                # In this mode, we only compute KL divergence
              
                _, _, kl_sep, tc_loss, mi_loss = compute_tc_vae_loss(
                    pos_pred=pos_pred,
                    pos_true=data.pos,
                    edge_index=data.edge_index, 
                    mean=mean, 
                    logvar=log_var, 
                    batch=data.batch, 
                    beta=beta
                    )
                kl_loss = kl_sep + tc_loss + mi_loss
                recon_loss = torch.tensor(0)

            else:
                total_loss, recon_loss, kl_sep, tc_loss, mi_loss = compute_tc_vae_loss(
                    pos_pred=pos_pred,
                    pos_true=data.pos,
                    edge_index=data.edge_index, 
                    mean=mean, 
                    logvar=log_var, 
                    batch=data.batch, 
                    beta=beta,
                    tc_weight=tc_weight,
                    mi_weight=mi_weight,
                    dkl_weight=dkl_weight
                    )
            
            #Check if the losses are finite
            if not torch.isfinite(total_loss):
                print(f"Warning: tc_vae_loss is not finite. exiting the training loop.")
                print(f"tc_vae_loss: {total_loss}, recon_loss: {recon_loss}, kl_sep: {kl_sep}, tc_loss: {tc_loss}, mi_loss: {mi_loss}")
                break   
            if not torch.isfinite(kl_sep):
                print(f"Warning: kl_sep is not finite. exiting the training loop.")
                print(f"tc_vae_loss: {total_loss}, recon_loss: {recon_loss}, kl_sep: {kl_sep}, tc_loss: {tc_loss}, mi_loss: {mi_loss}")
                break
            if not torch.isfinite(tc_loss):
                print(f"Warning: tc_loss is not finite. exiting the training loop.")
                print(f"tc_vae_loss: {total_loss}, recon_loss: {recon_loss}, kl_sep: {kl_sep}, tc_loss: {tc_loss}, mi_loss: {mi_loss}")
                break
            if not torch.isfinite(mi_loss):
                print(f"Warning: mi_loss is not finite. exiting the training loop.")
                print(f"tc_vae_loss: {total_loss}, recon_loss: {recon_loss}, kl_sep: {kl_sep}, tc_loss: {tc_loss}, mi_loss: {mi_loss}")
                break
            if not torch.isfinite(recon_loss):
                print(f"Warning: recon_loss is not finite. exiting the training loop.")
                print(f"tc_vae_loss: {total_loss}, recon_loss: {recon_loss}, kl_sep: {kl_sep}, tc_loss: {tc_loss}, mi_loss: {mi_loss}")
                break


        if USE_FORCE_FIELD and lambda_energy > 0:
            # 2. Compute improved physics loss
            if SCALE_POSITIONS:
                rescaled_pred_coords = pos_pred * max_positions
            else:
                rescaled_pred_coords = pos_pred

            phys_loss, bond_e, angle_e, lj_e = physics_loss(
                physics_critic, 
                rescaled_pred_coords, 
                data.batch, 
                use_log=USE_LOG_FF,
                use_bonded=USE_BOND_FF,
                use_angle=USE_ANGLE_FF,
                use_lj=USE_LJ_FF) 

            total_loss = total_loss + lambda_energy * phys_loss

       

       
        total_loss.backward()

        mean_grad = model.encoder.to_latent_mean.weight.grad # keep for tracking exploding gradients
        if weight_mean_debug or debug:
           
            if mean_grad is not None:
                print(f"  -> Mean Grad Norm: {mean_grad.norm().item():.6f}")
            else:
                print("  -> Mean Grad is None!")




         # Clip gradients
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=False)

        optimizer.step()

        if not advanced_kl_loss:
            # set zero tensors 
            tc_loss = torch.tensor(0.0, device=device)
            mi_loss = torch.tensor(0.0, device=device)
            kl_sep = kl_loss  # In this case, kl_sep is the same as kl_loss

        # Accumulate losses (ensure all are scalars)
        train_loss += total_loss.item() 
        if advanced_kl_loss:
            train_tc_loss += tc_loss.item()
            train_mi_loss += mi_loss.item()
        
        train_kl_sep_loss += kl_sep.item()
          
        
       
        train_recon_loss += recon_loss.item()
        if USE_FORCE_FIELD and lambda_energy > 0:
            train_force_loss += phys_loss.item()


        kl_item = kl_loss.item() if not advanced_kl_loss else kl_sep.item()

        # Update progress bar
        if USE_FORCE_FIELD and lambda_energy > 0:
            train_pbar.set_postfix(
                loss=total_loss.item(),
                recon_loss=recon_loss.item(),
                kl_loss=kl_item,
                # mi_loss=mi_loss.item(),
                # tc_loss=tc_loss.item(),
                mean=torch.mean(mean).item(), 
                logvar=torch.mean(log_var).item(), 
                force_loss=phys_loss.item(),
                beta=beta,
                lambda_energy=lambda_energy               
            )
        else:
            train_pbar.set_postfix(
                mean_grad=mean_grad.norm().item(),
                loss=total_loss.item(),
                recon_loss=recon_loss.item(),
                kl_loss=kl_sep.item(),
                # mi_loss=mi_loss.item(),
                # tc_loss=tc_loss.item(),
                mean=torch.mean(mean).item(),
                logvar=torch.mean(log_var).item(),
                beta=beta
               
            )

    train_pbar.close()
  
    # Average the losses
    train_loss /= len(train_loader)
    
    train_kl_sep_loss /= len(train_loader)
    train_recon_loss /= len(train_loader)
    train_mi_loss /= len(train_loader)
    train_tc_loss /= len(train_loader)

    if USE_FORCE_FIELD and lambda_energy > 0:
        train_force_loss /= len(train_loader)
        print(f"Epoch {epoch+1}/{STARTING_EPOCH+EPOCHS}, Train Loss: {train_loss:.4f}, KL Loss: {train_kl_sep_loss:.2e}, TC_loss: {train_tc_loss:.2e}, MI_loss: {train_mi_loss:.2e}, Recon Loss: {train_recon_loss:.2e}, Force Loss: {train_force_loss:.2e}")
        if ENCODER_TYPE == 'denoise':
            print(f"Noise Level: {noise:.4f}")
    else:
        print(f"Epoch {epoch+1}/{STARTING_EPOCH+EPOCHS}, Train Loss: {train_loss:.4f}, KL Loss: {train_kl_sep_loss:.2e}, TC_loss: {train_tc_loss:.2e}, MI_loss: {train_mi_loss:.2e}, Recon Loss: {train_recon_loss:.2e}")

    if USE_SCHEDULER:
        scheduler.step(train_loss)


    # VALIDATION - Fixed to match training logic
    model.eval()
    val_loss = 0
    val_kl_loss = 0
    val_recon_loss = 0
    if USE_FORCE_FIELD:
        val_force_loss = 0

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)

             
            if TRAINING_MODE == 'denoising':
                # Create a noisy version of the BATCH's target positions
                pos_ref_for_decoder = data.pos + torch.randn_like(data.pos) * NOISE_LEVEL
            elif TRAINING_MODE == 'generative' and pos_ref is not None:
            # Use the single, averaged reference for the whole batch
                num_graphs_in_batch = data.batch.max().item() + 1
                pos_ref_for_decoder = pos_ref.repeat(num_graphs_in_batch, 1)
            elif TRAINING_MODE == 'generative' and pos_ref is None:
                pos_ref_for_decoder = None
            else:
                raise ValueError(f"Unknown TRAINING_MODE: {TRAINING_MODE}")
            #pos_ref = val_loader.dataset[np.random.randint(len(val_loader.dataset))].pos.to(device)
            pos_pred, mean, log_var, batch_vec = model(data, pos_ref=pos_ref_for_decoder)
            
            # Basic losses
            kl_loss = KL_divergence(mean, log_var)
            if advanced_recon_loss:
                recon_loss = advanced_reconstruction_loss(pos_pred, data.pos, data.edge_index, data.batch, align_coords=ALIGN_RECONS_LOSS) 
            else:
                recon_loss = reconstruction_loss(pos_pred, data.pos, data.batch, align=ALIGN_RECONS_LOSS)

            # Base validation loss
            if kl_loss < MIN_KL:
                total_loss = recon_loss
            else:
                total_loss = recon_loss + beta * kl_loss

            # Physics loss for validation (consistent with training)
            physics_loss_val = torch.tensor(0.0, device=device)
            if USE_FORCE_FIELD and lambda_energy > 0:
                try:
                    if SCALE_POSITIONS:
                        rescaled_pred_coords = pos_pred * max_positions 
                    else:
                        rescaled_pred_coords = pos_pred

                    physics_los_vals, bond_e, angle_e, lj_e = physics_loss(
                            physics_critic, 
                            rescaled_pred_coords, 
                            data.batch, 
                            use_log=USE_LOG_FF,
                            use_bonded=USE_BOND_FF,
                            use_angle=USE_ANGLE_FF,
                            use_lj=USE_LJ_FF) 

                    if not physics_loss_val.isnan().any():
                        total_loss += lambda_energy * physics_loss_val

                except Exception as e:
                    print(f"Error in validation physics loss: {e}")
                    physics_loss_val = torch.tensor(0.0, device=device)

            # Accumulate validation losses
            val_loss += total_loss.item()
            val_kl_loss += kl_loss.item()
            val_recon_loss += recon_loss.item()
            if USE_FORCE_FIELD and lambda_energy > 0:
                val_force_loss += physics_loss_val #.item()

    # Average validation losses
    val_loss /= len(val_loader)
    val_kl_loss /= len(val_loader)
    val_recon_loss /= len(val_loader)
    if USE_FORCE_FIELD and lambda_energy > 0:
        val_force_loss /= len(val_loader)

    print(f"Epoch {epoch+1}/{STARTING_EPOCH+EPOCHS}, Validation Loss: {val_loss:.2e}")

    # Log the results in TensorBoard
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Loss/KL_train', train_kl_loss, epoch)
    writer.add_scalar('Loss/KL_val', val_kl_loss, epoch)
    writer.add_scalar('Loss/Reconstruction_train', train_recon_loss, epoch)
    writer.add_scalar('Loss/Reconstruction_val', val_recon_loss, epoch)
    writer.add_scalar('Loss/TC_train', train_tc_loss, epoch)
    writer.add_scalar('Loss/MI_train', train_mi_loss, epoch)
    
    if USE_FORCE_FIELD and lambda_energy > 0:
        writer.add_scalar('Loss/Physics_train', train_force_loss, epoch)
        writer.add_scalar('Loss/Physics_val', val_force_loss, epoch)
        writer.add_scalar('Lambda', lambda_energy, epoch)
    
    writer.add_scalar('Beta', beta, epoch)
    if USE_SCHEDULER:
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
    else:
        writer.add_scalar('Learning Rate', LEARNING_RATE, epoch)

    # save the model every 30 epochs
    if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
        torch.save(model.state_dict(), file_path + f'model_epoch_{epoch+1}.pth')
        if verbose: print(f"Model saved to {file_path}model_epoch_{epoch+1}.pth")

    # === MODIFIED TEST/ANALYSIS BLOCK ===
    # Run analysis periodically or at the end of training
    if TEST_MODEL and ((epoch + 1) % 5== 0 or epoch == EPOCHS - 1):
        ep = epoch + 1
        run_full_analysis(model, test_loader, device, file_path, config, physics_critic, pos_ref, max_positions, ep, verbose=verbose)