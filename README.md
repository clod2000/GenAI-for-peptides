# GenAI-for-peptides

This repository contains the code and resources for my master's thesis, which explores generative artificial intelligence and geometric deep learning for peptide molecular data. The project focuses on graph-based neural architectures for learning, generating, and analyzing peptide structures, integrating physical chemistry principles via classical force fields.

---

## Project Overview

The main objective is to develop and investigate Variational Autoencoders (VAEs) enhanced with geometric neural networks (especially EGNNs) for learning and generating peptide structures from molecular dynamics (MD) simulations. A core innovation is the integration of differentiable physics-based loss functions (force fields), enforcing chemical realism in generated conformations. Both full-atom and dihedral-angle-based models are supported.

---

## Features

- **Automated conversion of MD trajectories into full-atom graph representations** via customizable scripts.
- **Full-atom graph-based and dihedral-angle-based modeling of peptides.**
- **Flexible VAE architectures:** original and hybrid displacement versions.
- **Trajectory data preprocessing, feature scaling, and coordinate alignment.**
- **Customizable training scripts and configuration templates.**
- **Visualization utilities** for molecular graph data and generated structures.
- **Differentiable force field integration:** bond, angle, Lennard-Jones (and advanced nonbonded terms).
- **Batch-wise physics loss** for physically meaningful regularization.
- **Analysis suite:** for reconstruction quality, latent space visualization, generation, and interpolation.

---

## Project Structure

```
GenAI-for-peptides/
├── FULL_ATOM/
│   └── CODES/
│       ├── fmain_analysis.py         # Training and analysis with force field loss
│       ├── test.sh                   # Bash script for test runs with different hyperparameters
│       ├── config.template.in        # Template config for experiments
│       ├── LIBS/
│           ├── FGVAE.py              # VAE model definition with EGNN encoder/decoder
│           ├── egnn_clean.py         # EGNN layer implementation (adapted from original source)
│           ├── utils.py              # Data handling, preprocessing, loss, and visualization
│           ├── create_full_graph_data.py # Graph data preprocessing and visualization
│           ├── force_field.py        # Differentiable force field module (energy loss)
│           ├── upgraded_ff.py        # Advanced force field with Coulomb and 1-4 exceptions
│           ├── old_force_field.py    # Legacy/alternate force field implementations
│           └── weight_check.py       # Model health diagnostics
|
|
├── DIHEDRALS/                        # Simpler GVAE for dihedral angles (see below)
│   ├── CODES/
│       ├── dmain.py                  # Main script for dihedral-based VAE training
│       ├── LIBS/
│       │   ├── DGVAE.py              # Dihedral VAE model definition
│       │   ├── dutils.py     # Utility functions for dihedral models
|
|
├── PIGVAE/                        # Experimental version to address problems encountered
                                   # in FULL_ATOM (please ignore it for now)

```

---

## Main Components

### Full-Atom Modeling

- **`fmain_analysis.py`**: Main script for configuring, training, and evaluating the full-atom VAE models. Support both standard and physics-regularized training modes.
- **`LIBS/` directory**: Contains EGNN layers, VAE architecture, data utilities, advanced loss functions, visualization, and force field modules.

### Dihedral Angle Modeling

The DIHEDRALS module provides a streamlined alternative for learning on molecular dihedral angles (torsions), which are lower-dimensional but critical for peptide backbone conformation. This module is particularly suitable for fast prototyping, coarse-grained studies, or as a baseline.

- **`DIHEDRALS/CODES/dmain.py`**: Main script for training a Graph VAE on dihedral angle data.
- **`DGVAE.py`**: Model definition for dihedral-based VAE.
- **`dutils.py`**: Data preprocessing, reconstruction, and loss functions specific to dihedral angles.

**Features of Dihedral Modeling:**
- Converts MD trajectories to sequences of backbone dihedral angles (e.g., phi/psi/omega).
- Builds simplified graph representations where nodes are residues and edges reflect peptide connectivity.
- Loss functions measure angular reconstruction accuracy (with periodicity handling).
- Fast training and inference due to reduced dimensionality.
- Visualization and analysis tools for angle distributions, latent space, and generative sampling.
- Can be used standalone or as a complement to full-atom modeling.

---

## Force Field Integration

This project integrates a differentiable classical force field (FF) module to enhance molecular structure learning and enforce physical plausibility of generated peptides. The force field is implemented in [`LIBS/force_field.py`](FULL_ATOM/CODES/LIBS/force_field.py) and optionally [`LIBS/upgraded_ff.py`](FULL_ATOM/CODES/LIBS/upgraded_ff.py), and is based on parameters extracted from standard molecular mechanics force fields via OpenMM (`amber99sb.xml`, `tip3p.xml`):

- **Bonded interactions**: Harmonic bond stretching and angle bending.
- **Nonbonded interactions**: Lennard-Jones (van der Waals) potential.
- **(Advanced, optional)**: Coulomb interactions and 1-4 scaling via upgraded_ff.

**Features:**

- **OpenMM-based parameter extraction**: Loads all relevant parameters from a reference PDB structure.
- **Differentiable energy loss**: Computes physics-based energy terms for each structure in a batch, integrating with PyTorch autograd.
- **Configurable loss weighting**: Select which energy terms to include and their weights via config (e.g. enable/disable bond, angle, LJ).
- **Annealing and scheduling**: Physics loss weight can be annealed during training to gradually enforce physical constraints.
- **Energy calculation for analysis**: Direct comparison between generated and real structures using OpenMM reference energies.

**Usage:**  
Enable force field regularization by setting `USE_FORCE_FIELD: True` in your config file, and provide a suitable PDB file via `PDB_FOR_ENERGY`. You may configure which energy terms to include (bond, angle, LJ) and their weights.

Example config snippet:
```
USE_FORCE_FIELD: True
PDB_FOR_ENERGY: ../DATA/raw/protein_only.pdb
USE_BOND_FF: True
USE_ANGLE_FF: True
USE_LJ_FF: True
LAMBDA_ENERGY: 0.001
```

---

## Data Availability

The molecular dynamics datasets used in this project are generated from GROMACS simulations of tetraaline molecules brought to equilibrium. Simulation trajectories (typically `.tpr` and `.xtc`) are not included in the repo due to size, but scripts are provided to convert compatible GROMACS output to graph datasets (full-atom and dihedral).

If you wish to use your own data, ensure your MD simulations are compatible with the preprocessing scripts, which expect GROMACS trajectory and topology file formats.

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- PyTorch Geometric
- MDAnalysis
- OpenMM (for force field module)
- Additional libraries as required by the code

You may need to install extra dependencies via `pip` or `conda`, especially for molecular simulation and analysis.

### Example Usage

1. **Clone the repository:**
    ```sh
    git clone https://github.com/clod2000/GenAI-for-peptides.git
    cd GenAI-for-peptides/FULL_ATOM/CODES
    ```

2. **Prepare your configuration file**, or use the provided template and modify as needed.

3. **Run a single experiment (full-atom):**
    ```sh
    python fmain.py --config config.template.in
    ```

4. **For dihedral experiments:**
    ```sh
    cd ../../DIHEDRALS/CODES
    python dmain.py --config config.dihedral.in
    ```

5. **For batch experiments, use the provided shell scripts:**
    ```sh
    bash single_sim.sh
    ```

---

## Analysis Suite

After training, the analysis routines provide:

- **Reconstruction quality**: RMSD, angular error, visual overlays, and energy comparison for best/worst reconstructions.
- **Latent space visualization**: PCA plots colored by physical energy or angle statistics.
- **Generative sampling**: Generation of new structures, energy/angle distribution comparison, and PDB export.
- **Interpolation**: Smooth interpolation in latent space with trajectory and GIF export.

See `fmain_analysis.py`, `LIBS/utils.py`, and DIHEDRALS analysis scripts for details.

---

## Current Status

This project is under active development as part of my master's thesis. Expect frequent changes and refactoring.

---

## Author

Claudio Colturi

---

_This repository is part of my master's thesis and is a work in progress. Contributions, suggestions, and feedback are welcome!_
