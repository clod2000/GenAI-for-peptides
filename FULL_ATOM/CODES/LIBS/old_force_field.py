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
        
#         # We don't need a full-blown integrator for single-point energy
#         self.integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        
#         # --- START OF REFACTORED PLATFORM AND CONTEXT INITIALIZATION ---
        
#         self.context = None
#         self.platform = None
        
#         # Define platform configurations to try in order of preference
#         platforms_to_try = []
#         if prefer_gpu:
#             platforms_to_try.extend([
#                 ('CUDA', {'CudaPrecision': 'mixed'}),
#                 ('CUDA', {'CudaPrecision': 'single'}),
#                 ('OpenCL', {'OpenCLPrecision': 'mixed'}),
#                 ('OpenCL', {'OpenCLPrecision': 'single'}),
#             ])
#         # Always add CPU as the final fallback
#         platforms_to_try.append(('CPU', {}))

#         # Loop through the configurations and try to create a context
#         for platform_name, properties in platforms_to_try:
#             try:
#                 platform = mm.Platform.getPlatformByName(platform_name)
#                 # Attempt to create the final context directly.
#                 # If this fails, the exception will be caught and we'll try the next config.
#                 self.context = mm.Context(self.system, self.integrator, platform, properties)
#                 self.platform = platform
#                 # If we succeed, break the loop
#                 print(f"Successfully created OpenMM context on: {platform.getName()} with properties: {properties}")
#                 break
#             except Exception as e:
#                 print(f"Failed to create context on {platform_name} with {properties}. Reason: {e}")
#                 self.context = None # Ensure context is None if creation failed
        
#         # If after all attempts, context is still None, something is seriously wrong.
#         if self.context is None:
#             raise RuntimeError("Could not create an OpenMM context on any available platform.")
        
    
#         # Print additional info for GPU platforms
#         if self.platform.getName() in ['CUDA', 'OpenCL']:
#             self._print_gpu_info()


#         # extract the parameters for the harmonic potential over bonded interactions

#         # Get HarmonicBondForce object
#         for force in self.system.getForces():
#             if isinstance(force, HarmonicBondForce):
#                 bond_force = force
#                 break

#         # Extract bond parameters
#         bonds = []
#         k_list = []
#         r0_list = []

#         for i in range(bond_force.getNumBonds()):
#             p1, p2, length, k = bond_force.getBondParameters(i)
#             bonds.append((p1, p2))
#             r0_list.append(length.value_in_unit_system(unit.md_unit_system))
#             k_list.append(k.value_in_unit_system(unit.md_unit_system))

#         self.bonds = bonds
#         self.r0_list = r0_list
#         self.k_list = k_list

#         # Extract Lennard-Jones parameters from NonbondedForce

#         for force in self.system.getForces():
#             if isinstance(force, NonbondedForce):
#                 nbforce = force
#                 break

#         sigma_list = []
#         epsilon_list = []

#         for i in range(nbforce.getNumParticles()):
#             charge, sigma, epsilon = nbforce.getParticleParameters(i)
#             sigma_list.append(sigma.value_in_unit_system(unit.md_unit_system))
#             epsilon_list.append(epsilon.value_in_unit_system(unit.md_unit_system))

#         self.sigma_list = sigma_list
#         self.epsilon_list = epsilon_list

#         # Extract angle parameters 
#         for force in self.system.getForces():
#             if isinstance(force, HarmonicAngleForce):
#                 angle_force = force
#                 break

#         # Extract indices and parameters
#         angle_indices = []
#         angle_params = []

#         for i in range(angle_force.getNumAngles()):
#             a1, a2, a3, theta0, k = angle_force.getAngleParameters(i)
#             angle_indices.append([a1, a2, a3])
#             angle_params.append((
#                 theta0.value_in_unit(unit.radian),  # equilibrium angle
#                 k.value_in_unit(unit.kilojoule_per_mole / unit.radian**2)  # force constant
#             ))

#         self.angle_indices = torch.tensor(angle_indices, dtype=torch.long)
#         self.angle_params = torch.tensor(angle_params, dtype=torch.float32)

#     def __del__(self):
#         """
#         Explicitly clean up the OpenMM context to prevent segfaults on exit.
#         """
#         # The 'hasattr' check is important in case initialization failed
#         if hasattr(self, 'context') and self.context is not None:
#             # We must delete the context first, then the integrator
#             # to release platform resources cleanly.
#             # In modern OpenMM, just unsetting the context reference is often enough.
#             del self.context
#             del self.integrator
#             # print("Cleaned up OpenMM context.") # Uncomment for debugging


#     def _print_gpu_info(self):
#         """Print GPU information for debugging."""
#         platform_name = self.platform.getName()
#         print(f"--- {platform_name} Platform Info ---")
#         for prop_name in self.platform.getPropertyNames():
#             try:
#                 value = self.context.getPlatform().getPropertyValue(self.context, prop_name)
#                 print(f"{prop_name}: {value}")
#             except Exception:
#                 # Some properties might not be queryable
#                 pass
#         print("--------------------------")

#     @torch.no_grad()
#     def openMM_energy(self, coords_tensor):
#         """
#         Calculates the potential energy for a given set of coordinates.
        
#         Args:
#             coords_tensor (torch.Tensor): A tensor of shape (num_atoms, 3).
#                                           Assumes coordinates are in Angstroms.
        
#         Returns:
#             torch.Tensor: A scalar tensor containing the potential energy in kJ/mol.
#         """
#         # Ensure the tensor is on the CPU and is a numpy array for OpenMM
#         coords_numpy = coords_tensor.detach().cpu().numpy()
#         positions = coords_numpy * unit.angstrom
        
#         self.context.setPositions(positions)
#         state = self.context.getState(getEnergy=True)
#         energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        
#         return torch.tensor(energy, dtype=torch.float32)
    
#     @staticmethod
#     def harmonic_bond_energy(r, r0, k):
#         """
#         Calculate the harmonic bond energy.

#         Args:
#             r (float): Current bond length.
#             r0 (float): Equilibrium bond length.
#             k (float): Force constant.
            
#         Returns:
#             float: Harmonic bond energy.
#         """
#         return 0.5 * k * (r - r0) ** 2

#     @staticmethod
#     def lj_energy(coords, sigma, epsilon, excluded_pairs=None, cutoff=10.0):
#         """
#         Vectorized Lennard-Jones energy computation.

#         Args:
#             coords: [N, 3] positions in Å (requires_grad=True)
#             sigma: [N] tensor of per-atom σ (Å)
#             epsilon: [N] tensor of per-atom ε (kJ/mol)
#             excluded_pairs: set of (i, j) tuples to exclude
#             cutoff: LJ cutoff distance in Å

#         Returns:
#             Scalar LJ energy
#         """
#         N = coords.shape[0]
        
#         # Create distance matrix [N, N]
#         rij = torch.cdist(coords, coords, p=2)  # pairwise distances

#         # Avoid division by zero (diag)
#         diag_mask = torch.eye(N, dtype=torch.bool, device=coords.device)
#         rij[diag_mask] = float('inf')

#         # Compute σ_ij and ε_ij using Lorentz-Berthelot combining rules
#         sigma_i = sigma.unsqueeze(0)  # [1, N]
#         sigma_j = sigma.unsqueeze(1)  # [N, 1]
#         sigma_ij = 0.5 * (sigma_i + sigma_j)  # [N, N]

#         epsilon_i = epsilon.unsqueeze(0)
#         epsilon_j = epsilon.unsqueeze(1)
#         epsilon_ij = torch.sqrt(epsilon_i * epsilon_j)

#         # Compute LJ terms
#         sr6 = (sigma_ij / rij) ** 6
#         sr12 = sr6 ** 2
#         E = 4 * epsilon_ij * (sr12 - sr6)

#         # Mask excluded pairs
#         if excluded_pairs is not None:
#             mask = torch.ones((N, N), dtype=torch.bool, device=coords.device)
#             for i, j in excluded_pairs:
#                 mask[i, j] = False
#                 mask[j, i] = False
#             E = E * mask

#         # Apply cutoff
#         if cutoff is not None:
#             E = E * (rij < cutoff)

#         # Sum over i < j only
#         energy = torch.triu(E, diagonal=1).sum()

#         return energy

#     @staticmethod
#     def angle_energy(coords, angle_indices, angle_params):
#         """
#         Computes harmonic angle energy.

#         Args:
#             coords: [N, 3] tensor of atomic positions (Å), requires_grad=True
#             angle_indices: [M, 3] LongTensor of indices (A, B, C)
#             angle_params: list of tuples [(theta0_rad, k_theta), ...] for each angle

#         Returns:
#             Scalar energy (kJ/mol)
#         """
#         idx_A = angle_indices[:, 0]
#         idx_B = angle_indices[:, 1]
#         idx_C = angle_indices[:, 2]

#         A = coords[idx_A]  # [M, 3]
#         B = coords[idx_B]
#         C = coords[idx_C]

#         # Vectors BA and BC
#         BA = A - B
#         BC = C - B

#         # Normalize
#         BA_norm = torch.nn.functional.normalize(BA, dim=1)
#         BC_norm = torch.nn.functional.normalize(BC, dim=1)

#         # Cosine of angle
#         cos_theta = (BA_norm * BC_norm).sum(dim=1).clamp(-1.0, 1.0)
#         theta = torch.acos(cos_theta)  # [M]

#         # Parameters
#         theta0 = angle_params[:, 0]  # [M]
#         k_theta = angle_params[:, 1]  # [M]
#         # theta0 = torch.tensor([p[0] for p in angle_params], device=coords.device)
#         # k_theta = torch.tensor([p[1] for p in angle_params], device=coords.device)

#         # Harmonic energy
#         energy = 0.5 * k_theta * (theta - theta0) ** 2
#         return energy.sum()



#     def __call__(self, coords_tensor):

#         E_bond = 0
#         E_angle = 0
#         E_lj = 0

#         # Calculate bond energy
#         for i, (p1, p2) in enumerate(self.bonds):
#             r_current = np.linalg.norm(coords_tensor[p1] - coords_tensor[p2])/10  # Convert to nm
#             r0 = self.r0_list[i]
#             k = self.k_list[i]
#             E_bond += self.harmonic_bond_energy(r_current, r0, k)

#         # Calculate Lennard-Jones energy
#         sigma_tensor = torch.tensor(self.sigma_list, dtype=torch.float32, device=coords_tensor.device)
#         epsilon_tensor = torch.tensor(self.epsilon_list, dtype=torch.float32, device=coords_tensor.device)
#         E_lj = self.lj_energy(coords_tensor, sigma_tensor, epsilon_tensor, excluded_pairs=self.bonds)


#         # Calculate angle energy
#         E_angle = self.angle_energy(coords_tensor, self.angle_indices, self.angle_params)

#         return E_bond, E_angle, E_lj
    


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
    
    def __del__(self):
        """Clean up OpenMM resources"""
        if hasattr(self, 'context') and self.context is not None:
            del self.context
        if hasattr(self, 'integrator') and self.integrator is not None:
            del self.integrator
    
    
    def harmonic_bond_energy(self, coords):
        """
        Calculate differentiable bond energy
        """
        bond_energy = torch.tensor(0.0, device=coords.device, dtype=coords.dtype)
        
        for i, (p1, p2) in enumerate(self.bonds):
            # Vector between atoms
            r_vec = coords[p1] - coords[p2]
            
            # Current bond length (in nm)
            r_current = torch.norm(r_vec) / 10.0  # Convert Å to nm
            
            # Get parameters
            r0 = self.r0_list[i]
            k = self.k_list[i]
            
            # Calculate energy
            bond_energy = bond_energy + 0.5 * k * (r_current - r0)**2
            
        return bond_energy
    
    def lennard_jones_energy(self, coords):
        """
        Calculate differentiable Lennard-Jones energy
        """
        # Move parameters to the same device as coords
        device = coords.device
        sigma = torch.tensor(self.sigma_list, dtype=coords.dtype, device=device)
        epsilon = torch.tensor(self.epsilon_list, dtype=coords.dtype, device=device)
        
        # Create distance matrix
        dists = torch.cdist(coords, coords)
        n = coords.shape[0]
        
        # Mask for valid interactions (exclude self-interactions)
        mask = ~torch.eye(n, dtype=bool, device=device)
        
        # Get sigma for each pair
        sigma_i = sigma.unsqueeze(0).expand(n, n)
        sigma_j = sigma.unsqueeze(1).expand(n, n)
        sigma_ij = 0.5 * (sigma_i + sigma_j)
        
        # Get epsilon for each pair
        eps_i = epsilon.unsqueeze(0).expand(n, n)
        eps_j = epsilon.unsqueeze(1).expand(n, n)
        eps_ij = torch.sqrt(eps_i * eps_j)
        
        # Avoid division by zero
        safe_dists = torch.clamp(dists, min=0.1) / 10.0  # Convert Å to nm
        
        # Calculate LJ energy
        ratio = sigma_ij / safe_dists
        ratio6 = torch.pow(ratio, 6)
        ratio12 = torch.pow(ratio6, 2)
        
        energy_matrix = 4 * eps_ij * (ratio12 - ratio6)
        energy_matrix = torch.where(mask, energy_matrix, torch.zeros_like(energy_matrix))
        
        # Sum upper triangle only to avoid double counting
        energy = torch.triu(energy_matrix, diagonal=1).sum()
        
        return energy
    
    def angle_energy(self, coords):
        """
        Calculate differentiable angle energy
        """
        if not self.has_angles:
            return torch.tensor(0.0, device=coords.device, dtype=coords.dtype)
        
        angle_energy = torch.tensor(0.0, device=coords.device, dtype=coords.dtype)
        
        # Move indices and parameters to the same device as coords
        indices = self.angle_indices.to(coords.device)
        params = self.angle_params.to(coords.dtype).to(coords.device)
        
        # Extract atom positions for each angle
        idx_A = indices[:, 0]
        idx_B = indices[:, 1]  # Central atom
        idx_C = indices[:, 2]
        
        # Get positions
        A = coords[idx_A]
        B = coords[idx_B]
        C = coords[idx_C]
        
        # Vectors from central atom
        BA = A - B
        BC = C - B
        
        # Normalize vectors
        BA_norm = torch.nn.functional.normalize(BA, dim=1, eps=1e-10)
        BC_norm = torch.nn.functional.normalize(BC, dim=1, eps=1e-10)
        
        # Calculate angle
        cos_angle = torch.sum(BA_norm * BC_norm, dim=1)
        cos_angle = torch.clamp(cos_angle, -0.99999, 0.99999)  # Avoid NaN in gradient
        angle = torch.acos(cos_angle)
        
        # Get parameters
        theta0 = params[:, 0]
        k_angle = params[:, 1]
        
        # Harmonic potential
        energy = 0.5 * k_angle * (angle - theta0)**2
        angle_energy = torch.sum(energy)
        
        return angle_energy
    
    def openMM_energy(self, coords_tensor):
        """
        Calculate energy using OpenMM (non-differentiable)
        """
        try:
            # Convert to numpy for OpenMM
            coords_numpy = coords_tensor.detach().cpu().numpy()
            positions = coords_numpy * unit.angstrom
            
            # Calculate energy
            self.context.setPositions(positions)
            state = self.context.getState(getEnergy=True)
            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            
            return torch.tensor(energy, dtype=torch.float32, device='cpu')
            
        except Exception as e:
            print(f"Error in OpenMM energy calculation: {e}")
            return torch.tensor(1e6, dtype=torch.float32, device='cpu')  # High energy on failure

    def __call__(self, coords_tensor):
        """
        Calculate all energy components in a differentiable way
        
        Args:
            coords_tensor: Coordinates tensor [num_atoms, 3]
            
        Returns:
            Tuple of (bond_energy, angle_energy, lj_energy)
        """
        try:
            # Move to the same device as input
            device = coords_tensor.device
            dtype = coords_tensor.dtype
            
            # Calculate bond energy
            E_bond = self.harmonic_bond_energy(coords_tensor)
            
            # Calculate angle energy
            E_angle = self.angle_energy(coords_tensor)
            
            # Calculate LJ energy
            E_lj = self.lennard_jones_energy(coords_tensor)
            
            return E_bond, E_angle, E_lj
            
        except Exception as e:
            print(f"Error in differentiable energy calculation: {e}")
            # Return default values on error
            zero = torch.tensor(0.0, device=coords_tensor.device)
            return zero, zero, zero
        
  