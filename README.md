# GenAI-for-peptides

This repository contains the code and resources for my master's thesis, which focuses on applying generative artificial intelligence and geometric deep learning to peptide molecular data. The project aims to advance the modeling and generation of peptide structures using modern machine learning techniques.

## Project Overview

The main objective of this project is to develop and explore Variational Autoencoders (VAEs) enhanced with geometric neural networks (notably EGNNs) for learning and generating peptide structures from molecular dynamics (MD) simulations.

## Features

- **Automated conversion of MD trajectories into full-atom graph representations** using customizable preprocessing scripts.
- **Full-atom graph-based modeling of peptides.**
- **Flexible VAE architectures** (including original and hybrid displacement versions).
- **Support for trajectory data preprocessing and feature scaling.**
- **Customizable training scripts with configuration templates.**
- **Visualization utilities for molecular graph data.**

## Project Structure

Below is a schematic overview of the main directories and files:
```
GenAI-for-peptides/
├── FULL_ATOM/
│   └── CODES/
│       ├── fmain.py                  # Main entry point for full-atom VAE training
│       ├── single_sim.sh             # Bash script for running simulation batches
│       ├── test.sh                   # Bash script for test runs with different hyperparameters
│       ├── config.template.in        # Template config for experiments
│       ├── notebook_hybrid.ipynb     # Reference Jupyter notebook
│       ├── LIBS/
│       │   ├── FGVAE.py              # VAE model definition with EGNN encoder
│       │   ├── egnn_clean.py         # EGNN layer implementation (adapted from original source)
│       │   ├── utils.py              # Data handling and utility functions
│       │   └── create_full_graph_data.py # Graph data preprocessing and visualization
│       └── configs/
│           └── test/
│               └── sim_lr_0.0001_layers_3_kl_min_0.01_latent_dim_64.in
├── DIHEDRALS/                        # Simpler GVAE for dihedral angles (code exists, pending cleanup and comments)
```

## Main Components

### `fmain.py`
The main script for configuring, training, and evaluating the VAE models. Parameters are set via a config file.

### `LIBS/`
This directory contains implementations for the EGNN layers, VAE architecture, and utility functions for data loading and processing.

> **Note**: The `egnn_clean.py` library is adapted from [EGNN by vgsatorras](https://github.com/vgsatorras/egnn/tree/3c079e7267dad0aa6443813ac1a12425c3717558).

#### `LIBS/create_full_graph_data.py`

This script is essential for converting MD simulation data (GROMACS `.tpr`/`.xtc`) into datasets of molecular graphs suitable for geometric deep learning:

- **TrajectoryDataset class**: Loads MD trajectories, selects atoms, computes static features (atom type, charge, etc.), and generates edge indices based on chemical bonds.
- **Automatic batching**: Processes each trajectory frame into a PyTorch Geometric graph object.
- **Visualization**: Functions for 2D/3D graph plotting with per-atom coloring, useful for reports or debugging.
- **Extensible feature engineering**: Easily add new atomic features.
- **CLI Example**: The script can be run directly to create datasets and plot sample graphs.

#### `LIBS/utils.py`

A companion module providing essential utilities for modeling:

- `get_dataset`: Loads, preprocesses, scales, and (optionally) aligns graph datasets; handles one-hot encoding of atom types and feature normalization.
- `find_rigid_alignment`: Implements the Kabsch algorithm for frame alignment (crucial for geometric losses).
- `get_dataloaders`: Splits datasets into training/validation/test sets and constructs PyTorch DataLoaders.
- `parse_config`: Reads experiment parameters from configuration files.
- **Loss functions**: Includes KL-divergence and advanced position reconstruction loss with batch alignment.
- **Visualization**: High-level plotting routines for datasets and model predictions (2D/3D).

### `notebook_hybrid.ipynb`
A Jupyter notebook for experiments and exploratory data analysis, demonstrating the use of the main library components.

### `config.template.in`
A template configuration file that defines the architecture and training parameters for experiments. Bash scripts use this template to generate configs for batch runs.

### `single_sim.sh` and `test.sh`
Shell scripts for automating hyperparameter sweeps and organizing output/logs.

## Data Availability

The molecular dynamics datasets used in this project are generated from GROMACS simulations of tetraaline molecules brought to equilibrium. These simulation trajectories (typically as `.tpr` and `.xtc` files) provide the structural data that is converted into graph representations for model training and evaluation.

If you wish to use your own data, ensure that your MD simulations are compatible with the preprocessing scripts, which expect GROMACS trajectory and topology file formats.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- PyTorch Geometric
- MDAnalysis
- Additional libraries as required by the code

You may need to install extra dependencies via `pip` or `conda`.

### Example Usage

1. **Clone the repository:**
    ```sh
    git clone https://github.com/clod2000/GenAI-for-peptides.git
    cd GenAI-for-peptides/FULL_ATOM/CODES
    ```

2. **Prepare your configuration file**, or use the provided template and modify as needed.

3. **Run a single experiment:**
    ```sh
    python fmain.py --config config.template.in
    ```

4. **For batch experiments, use the provided shell scripts:**
    ```sh
    bash single_sim.sh
    ```

## Current Status

This project is under active development as part of my master's thesis. Expect frequent changes and refactoring.

## Author

Claudio Colturi

---

> _This repository is part of my master's thesis and is a work in progress. Contributions, suggestions, and feedback are welcome!_
