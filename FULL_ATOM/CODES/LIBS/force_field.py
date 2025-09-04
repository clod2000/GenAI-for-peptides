
import torch
import openmm as mm
from openmm import app, unit
import numpy as np
from openmm import HarmonicBondForce, NonbondedForce, HarmonicAngleForce

class EnergyCalculator:
    def __init__(self, pdb_file, forcefield_files=['amber99sb.xml', 'tip3p.xml'],    
                 add_hydrogens=False, prefer_gpu=True):
        """
        Initializes the OpenMM system from a PDB file and force fields.
        """
        # Initialize OpenMM system as before
        self.pdb = app.PDBFile(pdb_file)
        self.forcefield = app.ForceField(*forcefield_files)
       
        # Add missing hydrogens if needed
        if add_hydrogens:
            print("Adding hydrogens to the structure...")
            modeller = app.Modeller(self.pdb.topology, self.pdb.positions)
            modeller.addHydrogens(self.forcefield)
            self.topology = modeller.topology
            self.positions = modeller.positions
        else:
            self.topology = self.pdb.topology
            self.positions = self.pdb.positions

        self.system = self.forcefield.createSystem(self.topology, nonbondedMethod=app.NoCutoff)
        
        # Platform selection
        self.integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        
        platforms_to_try = []
        if prefer_gpu:
            platforms_to_try.extend([
                ('CUDA', {'CudaPrecision': 'single'}),  # less memory, faster test
                ('CUDA', {'CudaPrecision': 'mixed'}),
                ('OpenCL', {'OpenCLPrecision': 'mixed'}),
            ])
        platforms_to_try.append(('CPU', {}))

        # Try platforms in order
        for platform_name, properties in platforms_to_try:
            try:
                platform = mm.Platform.getPlatformByName(platform_name)
                self.context = mm.Context(self.system, self.integrator, platform, properties)
                self.platform = platform
                print(f"Using OpenMM platform: {platform.getName()}")
                break
            except Exception as e:
                print(f"Failed to create context on {platform_name}: {e}")
                self.context = None
        
        if self.context is None:
            raise RuntimeError("Could not create an OpenMM context on any available platform.")
        
        # Extract bond parameters with better error handling
        self.bonds = []
        self.r0_list = []
        self.k_list = []
        
        for force in self.system.getForces():
            if isinstance(force, HarmonicBondForce):
                bond_force = force
                for i in range(bond_force.getNumBonds()):
                    try:
                        p1, p2, length, k = bond_force.getBondParameters(i)
                        self.bonds.append((p1, p2))
                        self.r0_list.append(length.value_in_unit_system(unit.md_unit_system))
                        self.k_list.append(k.value_in_unit_system(unit.md_unit_system))
                    except Exception as e:
                        print(f"Error extracting bond parameters: {e}")
                break
        
        # Extract Lennard-Jones parameters
        self.sigma_list = []
        self.epsilon_list = []
        
        for force in self.system.getForces():
            if isinstance(force, NonbondedForce):
                nbforce = force
                for i in range(nbforce.getNumParticles()):
                    try:
                        charge, sigma, epsilon = nbforce.getParticleParameters(i)
                        self.sigma_list.append(sigma.value_in_unit_system(unit.md_unit_system))
                        self.epsilon_list.append(epsilon.value_in_unit_system(unit.md_unit_system))
                    except Exception as e:
                        print(f"Error extracting LJ parameters: {e}")
                        # Use defaults if extraction fails
                        self.sigma_list.append(0.3)
                        self.epsilon_list.append(0.0)
                break
        
        # Extract angle parameters
        angle_indices = []
        angle_params = []
        
        for force in self.system.getForces():
            if isinstance(force, HarmonicAngleForce):
                angle_force = force
                for i in range(angle_force.getNumAngles()):
                    try:
                        a1, a2, a3, theta0, k = angle_force.getAngleParameters(i)
                        angle_indices.append([a1, a2, a3])
                        angle_params.append([
                            theta0.value_in_unit(unit.radian),
                            k.value_in_unit(unit.kilojoule_per_mole / unit.radian**2)
                        ])
                    except Exception as e:
                        print(f"Error extracting angle parameters: {e}")
                break
        
        # Handle empty parameter lists
        if not angle_indices:
            self.has_angles = False
            self.angle_indices = torch.zeros((0, 3), dtype=torch.long)
            self.angle_params = torch.zeros((0, 2), dtype=torch.float32)
        else:
            self.has_angles = True
            self.angle_indices = torch.tensor(angle_indices, dtype=torch.long)
            self.angle_params = torch.tensor(angle_params, dtype=torch.float32)

        # --- ADDED: precompute index tensors and parameter tensors for vectorization ---
        # Bonds
        self.bond_indices = torch.tensor(self.bonds, dtype=torch.long) if self.bonds else torch.zeros((0,2), dtype=torch.long)
        self.r0_tensor = torch.tensor(self.r0_list, dtype=torch.float32) if self.r0_list else torch.zeros(0, dtype=torch.float32)
        self.k_tensor = torch.tensor(self.k_list, dtype=torch.float32) if self.k_list else torch.zeros(0, dtype=torch.float32)
        # LJ
        self.sigma_tensor = torch.tensor(self.sigma_list, dtype=torch.float32) if self.sigma_list else torch.zeros(0, dtype=torch.float32)
        self.epsilon_tensor = torch.tensor(self.epsilon_list, dtype=torch.float32) if self.epsilon_list else torch.zeros(0, dtype=torch.float32)
        # Angles already handled above

    def __del__(self):
        """Clean up OpenMM resources"""
        if hasattr(self, 'context') and self.context is not None:
            del self.context
        if hasattr(self, 'integrator') and self.integrator is not None:
            del self.integrator

    def harmonic_bond_energy(self, coords):
        """
        Vectorized bond energy calculation
        coords must be in nm!
        """
        if self.bond_indices.shape[0] == 0:
            return torch.tensor(0.0, device=coords.device, dtype=coords.dtype)
        # Ensure on correct device
        bond_indices = self.bond_indices.to(coords.device)
        r0 = self.r0_tensor.to(coords.device)
        k = self.k_tensor.to(coords.device)

        pos1 = coords[bond_indices[:, 0]]
        pos2 = coords[bond_indices[:, 1]]
        lengths = torch.norm(pos1 - pos2, dim=1)
        energy = 0.5 * k * (lengths - r0) ** 2
        return energy.sum()

    def lennard_jones_energy(self, coords, cutoff=0.5, exclude_bonded=True):
        """
        Vectorized Lennard-Jones energy calculation
        
        Args:
            coords: Tensor of shape [num_atoms, 3] in nm
            cutoff: Distance cutoff in nm (default 1.0 nm)
            exclude_bonded: Whether to exclude bonded pairs from LJ calculation
        
        Returns:
            Total Lennard-Jones energy as a scalar tensor
        """
        if coords.shape[0] < 2:
            return torch.tensor(0.0, device=coords.device, dtype=coords.dtype)
        
        # Move parameters to correct device
        sigma = self.sigma_tensor.to(coords.device)
        epsilon = self.epsilon_tensor.to(coords.device)
        n = coords.shape[0]
        
        # Calculate all pairwise distances
        dists = torch.cdist(coords, coords)  # [n, n]
        
        # Create mask to exclude self-interactions
        mask = ~torch.eye(n, dtype=torch.bool, device=coords.device)
        
        # Exclude bonded pairs if requested
        if exclude_bonded and hasattr(self, 'bond_indices') and self.bond_indices.shape[0] > 0:
            bond_indices = self.bond_indices.to(coords.device)
            for i in range(bond_indices.shape[0]):
                idx1, idx2 = bond_indices[i, 0], bond_indices[i, 1]
                mask[idx1, idx2] = False
                mask[idx2, idx1] = False
        
        # Apply cutoff mask
        if cutoff > 0:
            cutoff_mask = dists <= cutoff
            mask = mask & cutoff_mask
        
        # Compute σ_ij and ε_ij using Lorentz-Berthelot combining rules
        sigma_i = sigma.unsqueeze(0).expand(n, n)  # [n, n]
        sigma_j = sigma.unsqueeze(1).expand(n, n)  # [n, n]
        sigma_ij = 0.5 * (sigma_i + sigma_j)
        
        eps_i = epsilon.unsqueeze(0).expand(n, n)
        eps_j = epsilon.unsqueeze(1).expand(n, n)
        eps_ij = torch.sqrt(eps_i * eps_j)
        
        # Avoid division by zero and numerical instability
        safe_dists = torch.clamp(dists, min=0.01)  # Minimum distance 0.01 nm
        
        # Calculate LJ potential: E = 4ε[(σ/r)^12 - (σ/r)^6]
        ratio = sigma_ij / safe_dists
        ratio6 = ratio ** 6
        ratio12 = ratio6 ** 2
        E = 4 * eps_ij * (ratio12 - ratio6)
        
        # Apply all masks
        E = torch.where(mask, E, torch.zeros_like(E))
        
        # Sum upper triangle only to avoid double counting
        energy = torch.triu(E, diagonal=1).sum()
        
        return energy
       

    def angle_energy(self, coords):
        """
        Vectorized angle energy calculation
        coords must be in nm!
        """
        if not self.has_angles or self.angle_indices.shape[0] == 0:
            return torch.tensor(0.0, device=coords.device, dtype=coords.dtype)
        idx = self.angle_indices.to(coords.device)
        params = self.angle_params.to(coords.device)
        A = coords[idx[:, 0]]
        B = coords[idx[:, 1]]
        C = coords[idx[:, 2]]
        BA = A - B
        BC = C - B
        BA_norm = torch.nn.functional.normalize(BA, dim=1, eps=1e-10)
        BC_norm = torch.nn.functional.normalize(BC, dim=1, eps=1e-10)
        cos_angle = torch.sum(BA_norm * BC_norm, dim=1).clamp(-0.99999, 0.99999)
        angle = torch.acos(cos_angle)
        theta0 = params[:, 0]
        k_angle = params[:, 1]
        energy = 0.5 * k_angle * (angle - theta0) ** 2
        return energy.sum()

    def openMM_energy(self, coords_tensor):
        """
        Calculate energy using OpenMM (non-differentiable)
        """
        try:
            # Convert to numpy for OpenMM
            coords_numpy = coords_tensor.detach().cpu().numpy()
            positions = coords_numpy * unit.nanometer
            
            # Calculate energy
            self.context.setPositions(positions)
            state = self.context.getState(getEnergy=True)
            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            
            return torch.tensor(energy, dtype=torch.float32, device='cpu')
            
        except Exception as e:
            print(f"Error in OpenMM energy calculation: {e}")
            return torch.tensor(1e6, dtype=torch.float32, device='cpu')  # High energy on failure

    def __call__(self, coords_tensor,use_bonded=True, use_angle=True, use_lj=True):
        """
        Calculate all energy components in a differentiable way
        Assumes coords_tensor is in nm!

        Args:
            coords_tensor: Coordinates tensor [num_atoms, 3]
            use_bonded: Whether to include bonded energy
            use_angle: Whether to include angle energy
            use_lj: Whether to include Lennard-Jones energy
        Returns:
            Tuple of (bond_energy, angle_energy, lj_energy)
        """
        try:
            device = coords_tensor.device
            E_bond = self.harmonic_bond_energy(coords_tensor) if use_bonded else 0
            E_angle = self.angle_energy(coords_tensor) if use_angle else 0
            E_lj = self.lennard_jones_energy(coords_tensor) if use_lj else 0
            return E_bond, E_angle, E_lj
        except Exception as e:
            print(f"Error in differentiable energy calculation: {e}")
            zero = torch.tensor(0.0, device=coords_tensor.device)
            return zero, zero, zero

def physics_loss(energy_calculator, pos_pred, batch, use_log =True,
                  use_bonded=True, use_angle=True, use_lj=True):
    """
    Calculate differentiable physics-based loss
    
    Args:
        energy_calculator: EnergyCalculator instance
        pos_pred: Predicted positions [num_atoms, 3]
        batch: Batch indices
        use_log: Whether to use log scaling on the energy
        use_bonded: Whether to include bonded energy
        use_angle: Whether to include angle energy
        use_lj: Whether to include Lennard-Jones energy
    
    Returns:
        Total physics loss
    """
    # Initialize loss
    total_loss = 0.0
    num_graphs = batch.max().item() + 1
    
    for i in range(num_graphs):
        # Get positions for this molecule
        mask = batch == i
        coords = pos_pred[mask]
        
        try:
            # Calculate energy components
            bond_energy, angle_energy, lj_energy = energy_calculator(coords, use_bonded=use_bonded, use_angle=use_angle, use_lj=use_lj)

            # Weight the components
            physics_energy = (
                5.0 * bond_energy + 
                2.0 * angle_energy + 
                0.01 * lj_energy  
            )
            
           
            total_loss = total_loss + physics_energy

        except Exception as e:
            print(f"Error in physics loss calculation: {e}")
            # Don't add anything on error
    
    # Normalize by number of molecules
    if num_graphs > 0:
        total_loss = total_loss / num_graphs
    if use_log:
        total_loss = torch.log10(total_loss )
    
    return total_loss