# GenAI-for-peptides

This repository contains the code and resources for my master's thesis, focused on the application of generative artificial intelligence and geometric deep learning to peptide molecular data. The project is a work in progress and is organized in a modular and well-structured way to enable flexibility and extensibility for future developments.

## Project Overview

The main objective of this project is to develop and explore Variational Autoencoders (VAEs) enhanced with geometric neural networks (notably EGNNs) for learning and generating peptide structures from molecular dynamics data. The codebase is built with a strong emphasis on modularity, making it suitable for rapid prototyping and experimentation.

## Features

- **Full-atom graph-based representation of peptides.**
- **Flexible VAE architectures** (original and hybrid displacement).
- **Support for trajectory data preprocessing and feature scaling.**
- **Customizable training scripts with configuration templating.**
- **Visualization utilities for molecular graph data.**

## Project Structure

Below is a schematic tree of the main directories and files:
```
GenAI-for-peptides/
└── FULL_ATOM/
    └── CODES/
        ├── fmain.py                # Main entry point for full-atom VAE training
        ├── single_sim.sh           # Bash script for running simulation batches
        ├── test.sh                 # Bash script for test runs with different hyperparameters
        ├── config.template.in      # Template config for experiments
        ├── notebook_hybrid.ipynb   # Reference Jupyter notebook
        ├── LIBS/
        │   ├── FGVAE.py            # VAE model definition with EGNN encoder
        │   ├── egnn_clean.py       # EGNN layer implementation (adapted from original source)
        │   ├── utils.py            # Data handling and utility functions
        │   └── create_full_graph_data.py # Graph data preprocessing and visualization
        └── configs/
            └── test/
                └── sim_lr_0.0001_layers_3_kl_min_0.01_latent_dim_64.in
 └── DIHEDRALS/                     # Version of a simpler GVAE that captures only dihedral angles ( code already exists, still need to be cleaned and commented)
```


## Main Components

### `fmain.py`
The main script for configuring, training, and evaluating the VAE models. Parameters are set via a config file.

### `LIBS/`
A library directory holding implementation of the EGNN layers, VAE architecture, and utility functions for data loading and processing.
*NOTE*: the "egnn_clean.py" library is taken from [https://github.com/vgsatorras/egnn/blob/3c079e7267dad0aa6443813ac1a12425c3717558/models/egnn_clean/egnn_clean.py#L119](this link)

### `notebook_hybrid.ipynb`
A Jupyter notebook for experiments and exploratory data analysis, demonstrating the use of the main library components.

### `config.template.in`
A template configuration file that defines the architecture and training parameters for experiments. Bash scripts use this template to generate configs for batch runs.

### `single_sim.sh` and `test.sh`
Shell scripts for automating hyperparameter sweeps and organizing output/logs.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- PyTorch Geometric
- MDAnalysis
- Other dependencies as required in the code

You may need to install additional libraries via `pip` or `conda` as required.

### Example Usage

1. Clone the repository:
    ```sh
    git clone https://github.com/clod2000/GenAI-for-peptides.git
    cd GenAI-for-peptides/FULL_ATOM/CODES
    ```

2. Prepare your configuration file, or use the template and modify as needed.

3. Run a single experiment:
    ```sh
    python fmain.py --config config.template.in
    ```

4. For batch experiments, use the provided shell scripts:
    ```sh
    bash single_sim.sh
    ```

## Current Status

This project is in active development as part of my master's thesis. Expect frequent changes and refactoring as the work progresses.

## Author

Claudio Colturi

---

> _This repository is part of my master's thesis and is a work in progress. Contributions, suggestions, and feedback are welcome!_
