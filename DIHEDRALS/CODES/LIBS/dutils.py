import torch
import os
import numpy as np


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

