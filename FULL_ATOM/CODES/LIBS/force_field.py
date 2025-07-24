# import torch
# import openmm as mm
# from openmm import app, unit
# import numpy as np
# from openmm import HarmonicBondForce, NonbondedForce, HarmonicAngleForce

# class EnergyCalculator:
#     def __init__(self, pdb_file, forcefield_files=['amber99sb.xml', 'tip3p.xml'],    
#                  add_hydrogens=False, prefer_gpu=True):
#         """
#         Initializes the OpenMM system from a PDB file and force fields.
#         """
#         # Initialize OpenMM system as before
#         self.pdb = app.PDBFile(pdb_file)
#         self.forcefield = app.ForceField(*forcefield_files)
       
#         # Add missing hydrogens if needed
#         if add_hydrogens:
#             print("Adding hydrogens to the structure...")
#             modeller = app.Modeller(self.pdb.topology, self.pdb.positions)
#             modeller.addHydrogens(self.forcefield)
#             self.topology = modeller.topology
#             self.positions = modeller.positions
#         else:
#             self.topology = self.pdb.topology
#             self.positions = self.pdb.positions

#         self.system = self.forcefield.createSystem(self.topology, nonbondedMethod=app.NoCutoff)
        
#         # Platform selection
#         self.integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        
#         platforms_to_try = []
#         if prefer_gpu:
#             platforms_to_try.extend([
#                 ('CUDA', {'CudaPrecision': 'single'}),  # less memory, faster test
#                 ('CUDA', {'CudaPrecision': 'mixed'}),
#                 ('OpenCL', {'OpenCLPrecision': 'mixed'}),
#             ])
#         platforms_to_try.append(('CPU', {}))

#         # Try platforms in order
#         for platform_name, properties in platforms_to_try:
#             try:
#                 platform = mm.Platform.getPlatformByName(platform_name)
#                 self.context = mm.Context(self.system, self.integrator, platform, properties)
#                 self.platform = platform
#                 print(f"Using OpenMM platform: {platform.getName()}")
#                 break
#             except Exception as e:
#                 print(f"Failed to create context on {platform_name}: {e}")
#                 self.context = None
        
#         if self.context is None:
#             raise RuntimeError("Could not create an OpenMM context on any available platform.")
        
#         # Extract bond parameters with better error handling
#         self.bonds = []
#         self.r0_list = []
#         self.k_list = []
        
#         for force in self.system.getForces():
#             if isinstance(force, HarmonicBondForce):
#                 bond_force = force
#                 for i in range(bond_force.getNumBonds()):
#                     try:
#                         p1, p2, length, k = bond_force.getBondParameters(i)
#                         self.bonds.append((p1, p2))
#                         self.r0_list.append(length.value_in_unit_system(unit.md_unit_system))
#                         self.k_list.append(k.value_in_unit_system(unit.md_unit_system))
#                     except Exception as e:
#                         print(f"Error extracting bond parameters: {e}")
#                 break
        
#         # Extract Lennard-Jones parameters
#         self.sigma_list = []
#         self.epsilon_list = []
        
#         for force in self.system.getForces():
#             if isinstance(force, NonbondedForce):
#                 nbforce = force
#                 for i in range(nbforce.getNumParticles()):
#                     try:
#                         charge, sigma, epsilon = nbforce.getParticleParameters(i)
#                         self.sigma_list.append(sigma.value_in_unit_system(unit.md_unit_system))
#                         self.epsilon_list.append(epsilon.value_in_unit_system(unit.md_unit_system))
#                     except Exception as e:
#                         print(f"Error extracting LJ parameters: {e}")
#                         # Use defaults if extraction fails
#                         self.sigma_list.append(0.3)
#                         self.epsilon_list.append(0.0)
#                 break
        
#         # Extract angle parameters
#         angle_indices = []
#         angle_params = []
        
#         for force in self.system.getForces():
#             if isinstance(force, HarmonicAngleForce):
#                 angle_force = force
#                 for i in range(angle_force.getNumAngles()):
#                     try:
#                         a1, a2, a3, theta0, k = angle_force.getAngleParameters(i)
#                         angle_indices.append([a1, a2, a3])
#                         angle_params.append([
#                             theta0.value_in_unit(unit.radian),
#                             k.value_in_unit(unit.kilojoule_per_mole / unit.radian**2)
#                         ])
#                     except Exception as e:
#                         print(f"Error extracting angle parameters: {e}")
#                 break
        
#         # Handle empty parameter lists
#         if not angle_indices:
#             self.has_angles = False
#             self.angle_indices = torch.zeros((0, 3), dtype=torch.long)
#             self.angle_params = torch.zeros((0, 2), dtype=torch.float32)
#         else:
#             self.has_angles = True
#             self.angle_indices = torch.tensor(angle_indices, dtype=torch.long)
#             self.angle_params = torch.tensor(angle_params, dtype=torch.float32)
    
#     def __del__(self):
#         """Clean up OpenMM resources"""
#         if hasattr(self, 'context') and self.context is not None:
#             del self.context
#         if hasattr(self, 'integrator') and self.integrator is not None:
#             del self.integrator
    
    
#     def harmonic_bond_energy(self, coords):
#         """
#         Calculate differentiable bond energy
#         coords must be in nm!
#         """
#         bond_energy = torch.tensor(0.0, device=coords.device, dtype=coords.dtype)
        
#         for i, (p1, p2) in enumerate(self.bonds):
#             # Vector between atoms
#             r_vec = coords[p1] - coords[p2]
            
#             # Current bond length (in nm)
#             r_current = torch.norm(r_vec) # / 10.0  # Convert Å to nm
            
#             # Get parameters
#             r0 = self.r0_list[i]
#             k = self.k_list[i]
            
#             # Calculate energy
#             bond_energy = bond_energy + 0.5 * k * (r_current - r0)**2
            
#         return bond_energy
    
    
#     def lennard_jones_energy(self, coords):
#         """
#         Calculate differentiable Lennard-Jones energy
#         coords must be in nm!
#         """
#         # Move parameters to the same device as coords
#         device = coords.device
#         sigma = torch.tensor(self.sigma_list, dtype=coords.dtype, device=device)
#         epsilon = torch.tensor(self.epsilon_list, dtype=coords.dtype, device=device)
        
#         # Create distance matrix
#         dists = torch.cdist(coords, coords)
#         n = coords.shape[0]
        
#         # Mask for valid interactions (exclude self-interactions)
#         mask = ~torch.eye(n, dtype=bool, device=device)
        
#         # Get sigma for each pair
#         sigma_i = sigma.unsqueeze(0).expand(n, n)
#         sigma_j = sigma.unsqueeze(1).expand(n, n)
#         sigma_ij = 0.5 * (sigma_i + sigma_j)
        
#         # Get epsilon for each pair
#         eps_i = epsilon.unsqueeze(0).expand(n, n)
#         eps_j = epsilon.unsqueeze(1).expand(n, n)
#         eps_ij = torch.sqrt(eps_i * eps_j)
        
#         # Avoid division by zero
#         safe_dists = torch.clamp(dists, min=0.1)  #/ 10.0  # Convert Å to nm
        
#         # Calculate LJ energy
#         ratio = sigma_ij / safe_dists
#         ratio6 = torch.pow(ratio, 6)
#         ratio12 = torch.pow(ratio6, 2)
        
#         energy_matrix = 4 * eps_ij * (ratio12 - ratio6)
#         energy_matrix = torch.where(mask, energy_matrix, torch.zeros_like(energy_matrix))
        
#         # Sum upper triangle only to avoid double counting
#         energy = torch.triu(energy_matrix, diagonal=1).sum()
        
#         return energy
    
#     def angle_energy(self, coords):
#         """
#         Calculate differentiable angle energy
#         coords must be in nm!
#         """
#         if not self.has_angles:
#             return torch.tensor(0.0, device=coords.device, dtype=coords.dtype)
        
#         angle_energy = torch.tensor(0.0, device=coords.device, dtype=coords.dtype)
        
#         # Move indices and parameters to the same device as coords
#         indices = self.angle_indices.to(coords.device)
#         params = self.angle_params.to(coords.dtype).to(coords.device)
        
#         # Extract atom positions for each angle
#         idx_A = indices[:, 0]
#         idx_B = indices[:, 1]  # Central atom
#         idx_C = indices[:, 2]
        
#         # Get positions
#         A = coords[idx_A]
#         B = coords[idx_B]
#         C = coords[idx_C]
        
#         # Vectors from central atom
#         BA = A - B
#         BC = C - B
        
#         # Normalize vectors
#         BA_norm = torch.nn.functional.normalize(BA, dim=1, eps=1e-10)
#         BC_norm = torch.nn.functional.normalize(BC, dim=1, eps=1e-10)
        
#         # Calculate angle
#         cos_angle = torch.sum(BA_norm * BC_norm, dim=1)
#         cos_angle = torch.clamp(cos_angle, -0.99999, 0.99999)  # Avoid NaN in gradient
#         angle = torch.acos(cos_angle)
        
#         # Get parameters
#         theta0 = params[:, 0]
#         k_angle = params[:, 1]
        
#         # Harmonic potential
#         energy = 0.5 * k_angle * (angle - theta0)**2
#         angle_energy = torch.sum(energy)
        
#         return angle_energy
    
#     def openMM_energy(self, coords_tensor):
#         """
#         Calculate energy using OpenMM (non-differentiable)
#         """
#         try:
#             # Convert to numpy for OpenMM
#             coords_numpy = coords_tensor.detach().cpu().numpy()
#             positions = coords_numpy * unit.angstrom
            
#             # Calculate energy
#             self.context.setPositions(positions)
#             state = self.context.getState(getEnergy=True)
#             energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            
#             return torch.tensor(energy, dtype=torch.float32, device='cpu')
            
#         except Exception as e:
#             print(f"Error in OpenMM energy calculation: {e}")
#             return torch.tensor(1e6, dtype=torch.float32, device='cpu')  # High energy on failure

#     def __call__(self, coords_tensor):
#         """
#         Calculate all energy components in a differentiable way
        
#         Args:
#             coords_tensor: Coordinates tensor [num_atoms, 3]
            
#         Returns:
#             Tuple of (bond_energy, angle_energy, lj_energy)
#         """
#         try:
#             # Move to the same device as input
#             device = coords_tensor.device
#             dtype = coords_tensor.dtype
            
#             # Calculate bond energy
#             E_bond = self.harmonic_bond_energy(coords_tensor)
            
#             # Calculate angle energy
#             E_angle = self.angle_energy(coords_tensor)
            
#             # Calculate LJ energy
#             E_lj = self.lennard_jones_energy(coords_tensor)
            
#             return E_bond, E_angle, E_lj
            
#         except Exception as e:
#             print(f"Error in differentiable energy calculation: {e}")
#             # Return default values on error
#             zero = torch.tensor(0.0, device=coords_tensor.device)
#             return zero, zero, zero
        
  


# # import torch

# # from openmm import app, unit
# # from openmm import HarmonicBondForce, NonbondedForce, HarmonicAngleForce


# # print("Using OpenMM for force field extraction")


# # class EnergyCalculator:
# #     """Fully PyTorch-based force field with no OpenMM dependencies during forward/backward passes"""
    
# #     def __init__(self, pdb_file, forcefield_files=['amber99sb.xml', 'tip3p.xml']):
# #         """Extract parameters from OpenMM but store them as PyTorch tensors"""
# #         # Initialize OpenMM only for parameter extraction (on CPU)
# #         pdb = app.PDBFile(pdb_file)
# #         forcefield = app.ForceField(*forcefield_files)
# #         system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff)
        
# #         print("Extracting force field parameters from OpenMM...")
        
# #         # Extract bond parameters
# #         bonds = []
# #         r0_list = []
# #         k_list = []
        
# #         for force in system.getForces():
# #             if isinstance(force, HarmonicBondForce):
# #                 for i in range(force.getNumBonds()):
# #                     p1, p2, length, k = force.getBondParameters(i)
# #                     bonds.append((int(p1), int(p2)))
# #                     r0_list.append(length.value_in_unit(unit.nanometer))
# #                     k_list.append(k.value_in_unit(unit.kilojoule_per_mole/unit.nanometer**2))
        
# #         # Extract LJ parameters
# #         sigma_list = []
# #         epsilon_list = []
# #         charges = []
        
# #         for force in system.getForces():
# #             if isinstance(force, NonbondedForce):
# #                 for i in range(force.getNumParticles()):
# #                     charge, sigma, epsilon = force.getParticleParameters(i)
# #                     charges.append(charge.value_in_unit(unit.elementary_charge))
# #                     sigma_list.append(sigma.value_in_unit(unit.nanometer))
# #                     epsilon_list.append(epsilon.value_in_unit(unit.kilojoule_per_mole))
                
# #                 # Get excluded pairs
# #                 excluded_pairs = set()
# #                 for i in range(force.getNumExceptions()):
# #                     p1, p2, _, _, _ = force.getExceptionParameters(i)
# #                     excluded_pairs.add((min(p1, p2), max(p1, p2)))
        
# #         # Extract angle parameters
# #         angles = []
# #         theta0_list = []
# #         k_angle_list = []
        
# #         for force in system.getForces():
# #             if isinstance(force, HarmonicAngleForce):
# #                 for i in range(force.getNumAngles()):
# #                     a1, a2, a3, theta0, k = force.getAngleParameters(i)
# #                     angles.append((int(a1), int(a2), int(a3)))
# #                     theta0_list.append(theta0.value_in_unit(unit.radian))
# #                     k_angle_list.append(k.value_in_unit(unit.kilojoule_per_mole/unit.radian**2))
        
# #         # Store parameters as PyTorch tensors (CPU initially, will move to right device later)
# #         self.bonds = bonds
# #         self.r0_tensor = torch.tensor(r0_list, dtype=torch.float32)
# #         self.k_bond_tensor = torch.tensor(k_list, dtype=torch.float32)
        
# #         self.angles = angles
# #         self.theta0_tensor = torch.tensor(theta0_list, dtype=torch.float32)
# #         self.k_angle_tensor = torch.tensor(k_angle_list, dtype=torch.float32)
        
# #         self.sigma_tensor = torch.tensor(sigma_list, dtype=torch.float32)
# #         self.epsilon_tensor = torch.tensor(epsilon_list, dtype=torch.float32)
# #         self.charge_tensor = torch.tensor(charges, dtype=torch.float32)
# #         self.excluded_pairs = excluded_pairs
        
# #         self.num_atoms = len(sigma_list)
# #         print(f"Extracted parameters for {len(bonds)} bonds, {len(angles)} angles, and {self.num_atoms} atoms")
        
# #         # Clean up OpenMM objects - we don't need them anymore
# #         del system, pdb, forcefield
    
# #     def to(self, device):
# #         """Move all tensors to specified device"""
# #         self.r0_tensor = self.r0_tensor.to(device)
# #         self.k_bond_tensor = self.k_bond_tensor.to(device)
# #         self.theta0_tensor = self.theta0_tensor.to(device)
# #         self.k_angle_tensor = self.k_angle_tensor.to(device)
# #         self.sigma_tensor = self.sigma_tensor.to(device)
# #         self.epsilon_tensor = self.epsilon_tensor.to(device)
# #         self.charge_tensor = self.charge_tensor.to(device)
# #         return self
        
# #     def bond_energy(self, coords):
# #         """Calculate bond energy using PyTorch operations"""
# #         energy = torch.tensor(0.0, device=coords.device)
        
# #         for i, (atom1, atom2) in enumerate(self.bonds):
# #             # Get bond vector
# #             r_vec = coords[atom1] - coords[atom2]
            
# #             # Convert from Angstrom to nm
# #             r_current = torch.norm(r_vec) / 10.0
            
# #             # Get parameters
# #             r0 = self.r0_tensor[i]
# #             k = self.k_bond_tensor[i]
            
# #             # Harmonic potential: E = 0.5 * k * (r - r0)^2
# #             energy = energy + 0.5 * k * (r_current - r0)**2
            
# #         return energy
    
# #     def angle_energy(self, coords):
# #         """Calculate angle energy using PyTorch operations"""
# #         energy = torch.tensor(0.0, device=coords.device)
        
# #         for i, (atom1, atom2, atom3) in enumerate(self.angles):
# #             # Get positions (atom2 is central atom)
# #             p1 = coords[atom1]
# #             p2 = coords[atom2]
# #             p3 = coords[atom3]
            
# #             # Get vectors from central atom
# #             v1 = p1 - p2
# #             v2 = p3 - p2
            
# #             # Normalize vectors
# #             v1_norm = torch.nn.functional.normalize(v1, dim=0, eps=1e-10)
# #             v2_norm = torch.nn.functional.normalize(v2, dim=0, eps=1e-10)
            
# #             # Calculate angle
# #             cos_angle = torch.sum(v1_norm * v2_norm).clamp(-0.999999, 0.999999)
# #             angle = torch.acos(cos_angle)
            
# #             # Get parameters
# #             theta0 = self.theta0_tensor[i]
# #             k_angle = self.k_angle_tensor[i]
            
# #             # Harmonic potential: E = 0.5 * k * (theta - theta0)^2
# #             energy = energy + 0.5 * k_angle * (angle - theta0)**2
            
# #         return energy
    
# #     def lj_energy(self, coords, cutoff=1.0):
# #         """Calculate Lennard-Jones energy using PyTorch operations with a cutoff in nm"""
# #         energy = torch.tensor(0.0, device=coords.device)
        
# #         # Convert cutoff from nm to Angstrom
# #         cutoff_angstrom = cutoff * 10.0
        
# #         # Create all pairs of atoms
# #         for i in range(self.num_atoms):
# #             for j in range(i+1, self.num_atoms):
# #                 # Skip excluded pairs (typically bonded atoms)
# #                 if (min(i,j), max(i,j)) in self.excluded_pairs:
# #                     continue
                
# #                 # Calculate distance (in nm)
# #                 r_vec = coords[i] - coords[j]
# #                 r = torch.norm(r_vec) / 10.0  # Angstrom to nm
                
# #                 # Skip if beyond cutoff
# #                 if r > cutoff:
# #                     continue
                
# #                 # Get parameters with combining rules
# #                 sigma_i = self.sigma_tensor[i]
# #                 sigma_j = self.sigma_tensor[j]
# #                 sigma = 0.5 * (sigma_i + sigma_j)
                
# #                 epsilon_i = self.epsilon_tensor[i]
# #                 epsilon_j = self.epsilon_tensor[j]
# #                 epsilon = torch.sqrt(epsilon_i * epsilon_j)
                
# #                 # Lennard-Jones: E = 4ε[(σ/r)^12 - (σ/r)^6]
# #                 sr = sigma / torch.clamp(r, min=0.05)  # Avoid division by zero
# #                 sr6 = sr**6
# #                 sr12 = sr6**2
                
# #                 pair_energy = 4.0 * epsilon * (sr12 - sr6)
# #                 energy = energy + pair_energy
        
# #         return energy
    
# #     def __call__(self, coords):
# #         """Calculate all energy components for the given coordinates"""
# #         # Convert coordinates from Angstrom to nm if needed
# #         device = coords.device
        
# #         # Ensure parameters are on the same device
# #         if self.r0_tensor.device != device:
# #             self.to(device)
        
# #         # Calculate energies
# #         e_bond = self.bond_energy(coords)
# #         e_angle = self.angle_energy(coords)
# #         e_lj = self.lj_energy(coords)
        
# #         # Return individual components
# #         return e_bond, e_angle, e_lj


# def physics_loss(energy_calculator, pos_pred, batch):
#     """
#     Calculate differentiable physics-based loss
    
#     Args:
#         energy_calculator: EnergyCalculator instance
#         pos_pred: Predicted positions [num_atoms, 3]
#         batch: Batch indices
    
#     Returns:
#         Total physics loss
#     """
#     # Initialize loss
#     total_loss = 0.0
#     num_graphs = batch.max().item() + 1
    
#     for i in range(num_graphs):
#         # Get positions for this molecule
#         mask = batch == i
#         coords = pos_pred[mask]
        
#         try:
#             # Calculate energy components
#             bond_energy, angle_energy, lj_energy = energy_calculator(coords)
            
#             # Weight the components
#             physics_energy = (
#                 1.0 * bond_energy + 
#                 1.0 * angle_energy + 
#                 1.0 * lj_energy  
#             )
            
#             # Add to total
#             total_loss = total_loss + physics_energy
            
#         except Exception as e:
#             print(f"Error in physics loss calculation: {e}")
#             # Don't add anything on error
    
#     # Normalize by number of molecules
#     if num_graphs > 0:
#         total_loss = total_loss / num_graphs
    
#     return total_loss

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
        # [unchanged: resource cleanup]
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

    def lennard_jones_energy(self, coords):
        """
        Vectorized Lennard-Jones energy calculation
        coords must be in nm!
        """
        sigma = self.sigma_tensor.to(coords.device)
        epsilon = self.epsilon_tensor.to(coords.device)
        n = coords.shape[0]
        dists = torch.cdist(coords, coords)
        mask = ~torch.eye(n, dtype=torch.bool, device=coords.device)

        sigma_i = sigma.unsqueeze(0).expand(n, n)
        sigma_j = sigma.unsqueeze(1).expand(n, n)
        sigma_ij = 0.5 * (sigma_i + sigma_j)

        eps_i = epsilon.unsqueeze(0).expand(n, n)
        eps_j = epsilon.unsqueeze(1).expand(n, n)
        eps_ij = torch.sqrt(eps_i * eps_j)

        safe_dists = torch.clamp(dists, min=0.01)
        ratio = sigma_ij / safe_dists
        ratio6 = ratio ** 6
        ratio12 = ratio6 ** 2
        E = 4 * eps_ij * (ratio12 - ratio6)
        E = torch.where(mask, E, torch.zeros_like(E))
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
        # [unchanged: OpenMM energy using context]
        try:
            coords_numpy = coords_tensor.detach().cpu().numpy()
            positions = coords_numpy * unit.nanometer
            self.context.setPositions(positions)
            state = self.context.getState(getEnergy=True)
            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            return torch.tensor(energy, dtype=torch.float32, device='cpu')
        except Exception as e:
            print(f"Error in OpenMM energy calculation: {e}")
            return torch.tensor(1e6, dtype=torch.float32, device='cpu')

    def __call__(self, coords_tensor):
        """
        Calculate all energy components in a differentiable way
        Assumes coords_tensor is in nm!
        """
        try:
            device = coords_tensor.device
            E_bond = self.harmonic_bond_energy(coords_tensor)
            E_angle = self.angle_energy(coords_tensor)
            E_lj = self.lennard_jones_energy(coords_tensor)
            return E_bond, E_angle, E_lj
        except Exception as e:
            print(f"Error in differentiable energy calculation: {e}")
            zero = torch.tensor(0.0, device=coords_tensor.device)
            return zero, zero, zero

def physics_loss(energy_calculator, pos_pred, batch):
    """
    Calculate differentiable physics-based loss
    
    Args:
        energy_calculator: EnergyCalculator instance
        pos_pred: Predicted positions [num_atoms, 3]
        batch: Batch indices
    
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
            bond_energy, angle_energy, lj_energy = energy_calculator(coords)
            
            # Weight the components
            physics_energy = (
                5.0 * bond_energy + 
                2.0 * angle_energy + 
                0.01 * lj_energy  
            )
            
            # Add to total
            total_loss = total_loss + torch.log10(physics_energy + 1)

        except Exception as e:
            print(f"Error in physics loss calculation: {e}")
            # Don't add anything on error
    
    # Normalize by number of molecules
    if num_graphs > 0:
        total_loss = total_loss / num_graphs
    
    return total_loss