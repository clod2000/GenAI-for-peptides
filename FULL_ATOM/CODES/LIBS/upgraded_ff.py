import torch
import openmm as mm
from openmm import app, unit
import numpy as np
from openmm import HarmonicBondForce, NonbondedForce, HarmonicAngleForce

# Coulomb's constant in units for kJ/mol, elementary charge, and nm
COULOMB_CONSTANT = 138.935458

class EnergyCalculator:
    def __init__(self, pdb_file, forcefield_files=['amber99sb.xml', 'tip3p.xml'],
                 add_hydrogens=False, prefer_gpu=True): # Changed GPU default for easier debugging
        """
        Initializes the OpenMM system and extracts parameters to PyTorch tensors,
        including non-bonded exception parameters for 1-4 scaling.
        """
        self.pdb = app.PDBFile(pdb_file)
        self.forcefield = app.ForceField(*forcefield_files)
       
        if add_hydrogens:
            modeller = app.Modeller(self.pdb.topology, self.pdb.positions)
            modeller.addHydrogens(self.forcefield)
            self.topology, self.positions = modeller.topology, modeller.positions
        else:
            self.topology, self.positions = self.pdb.topology, self.pdb.positions

        # Use NoCutoff to ensure we calculate all interactions, matching our PyTorch goal
        self.system = self.forcefield.createSystem(self.topology, nonbondedMethod=app.NoCutoff)
        
        # --- Parameter Extraction ---
        
        # Standard parameters
        bonds, r0_list, k_list = [], [], []
        angle_indices, angle_params = [], []
        charges, sigma_list, epsilon_list = [], [], []
        
        # *** NEW: Store exception parameters ***
        self.exceptions = {}

        for force in self.system.getForces():
            if isinstance(force, HarmonicBondForce):
                for i in range(force.getNumBonds()):
                    p1, p2, length, k = force.getBondParameters(i)
                    bonds.append((p1, p2))
                    r0_list.append(length.value_in_unit_system(unit.md_unit_system))
                    k_list.append(k.value_in_unit_system(unit.md_unit_system))
            
            elif isinstance(force, HarmonicAngleForce):
                for i in range(force.getNumAngles()):
                    a1, a2, a3, theta0, k = force.getAngleParameters(i)
                    angle_indices.append([a1, a2, a3])
                    angle_params.append([theta0.value_in_unit(unit.radian), k.value_in_unit(unit.kilojoule_per_mole / unit.radian**2)])

            elif isinstance(force, NonbondedForce):
                force.setForceGroup(1)
                for i in range(force.getNumParticles()):
                    charge, sigma, epsilon = force.getParticleParameters(i)
                    charges.append(charge.value_in_unit(unit.elementary_charge))
                    sigma_list.append(sigma.value_in_unit_system(unit.md_unit_system))
                    epsilon_list.append(epsilon.value_in_unit_system(unit.md_unit_system))
                
                # *** NEW: Extract the exception parameters (1-2, 1-3, 1-4 pairs) ***
                for i in range(force.getNumExceptions()):
                    p1, p2, charge_prod, sigma, epsilon = force.getExceptionParameters(i)
                    key = tuple(sorted((p1, p2)))
                    self.exceptions[key] = {
                        'charge_prod': charge_prod.value_in_unit_system(unit.md_unit_system),
                        'sigma': sigma.value_in_unit_system(unit.md_unit_system),
                        'epsilon': epsilon.value_in_unit_system(unit.md_unit_system)
                    }

        # Store as tensors
        self.bond_indices = torch.tensor(bonds, dtype=torch.long) if bonds else torch.empty((0, 2), dtype=torch.long)
        self.r0_tensor = torch.tensor(r0_list, dtype=torch.float32)
        self.k_bond_tensor = torch.tensor(k_list, dtype=torch.float32)

        self.angle_indices = torch.tensor(angle_indices, dtype=torch.long) if angle_indices else torch.empty((0, 3), dtype=torch.long)
        self.angle_params = torch.tensor(angle_params, dtype=torch.float32)

        self.charge_tensor = torch.tensor(charges, dtype=torch.float32)
        self.sigma_tensor = torch.tensor(sigma_list, dtype=torch.float32)
        self.epsilon_tensor = torch.tensor(epsilon_list, dtype=torch.float32)

        # --- Context for OpenMM-based energy calculation ---
        # platform = mm.Platform.getPlatformByName('CPU')
        self.integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        #self.context = mm.Context(self.system, self.integrator, platform)
        #print(f"Using OpenMM platform for reference calculations: {platform.getName()}")

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

    def __del__(self):
        if hasattr(self, 'context'): del self.context
        if hasattr(self, 'integrator'): del self.integrator

    def nonbonded_energy(self, coords):
        """
        Vectorized Non-Bonded (LJ + Coulomb) energy that correctly handles
        1-2, 1-3, and 1-4 exceptions by building a scaling matrix.
        """
        n = coords.shape[0]
        if n < 2:
            return torch.tensor(0.0, device=coords.device, dtype=coords.dtype)
        
        device = coords.device
        
        # --- Pre-calculate full interaction matrices ---
        dists = torch.cdist(coords, coords, p=2)
        
        # Standard Lorentz-Berthelot combining rules
        sigma_ij = 0.5 * (self.sigma_tensor.to(device).unsqueeze(1) + self.sigma_tensor.to(device).unsqueeze(0))
        eps_ij = torch.sqrt(self.epsilon_tensor.to(device).unsqueeze(1) * self.epsilon_tensor.to(device).unsqueeze(0))
        
        # Charge products
        charge_prods = self.charge_tensor.to(device).unsqueeze(1) * self.charge_tensor.to(device).unsqueeze(0)
        
        # --- Build scaling matrices for exceptions ---
        # Start with all interactions included (scale = 1.0)
        lj_scaling = torch.ones((n, n), device=device)
        coulomb_scaling = torch.ones((n, n), device=device)

        for (p1, p2), params in self.exceptions.items():
            # Get the scaling factors from the exception parameters
            # If the standard charge_prod is non-zero, the scaling is exception_val / standard_val
            # If the standard charge_prod is zero, any non-zero exception value means a scaling of infinity,
            # but in practice, we just use the exception value directly.
            
            # For Coulomb
            std_q_prod = charge_prods[p1, p2]
            if abs(std_q_prod) > 1e-6:
                coulomb_scaling[p1, p2] = coulomb_scaling[p2, p1] = params['charge_prod'] / std_q_prod
            else:
                # This case is tricky, but usually means std was 0 and exception is also 0.
                # If exception were non-zero, it would be an override.
                coulomb_scaling[p1, p2] = coulomb_scaling[p2, p1] = 0.0
                
            # For LJ
            std_eps = eps_ij[p1, p2]
            if abs(std_eps) > 1e-6:
                lj_scaling[p1, p2] = lj_scaling[p2, p1] = params['epsilon'] / std_eps
            else:
                lj_scaling[p1, p2] = lj_scaling[p2, p1] = 0.0

        # --- Calculate energy using the scaling matrices ---
        safe_dists = torch.clamp(dists, min=1e-6)
        
        # LJ Term
        ratio = sigma_ij / safe_dists
        ratio6 = ratio ** 6
        lj_energy_matrix = 4 * eps_ij * (ratio6 ** 2 - ratio6)
        
        # Coulomb Term
        coulomb_energy_matrix = COULOMB_CONSTANT * charge_prods / safe_dists
        
        # Apply scaling
        total_energy_matrix = (lj_scaling * lj_energy_matrix) + (coulomb_scaling * coulomb_energy_matrix)
        
        # Mask for upper triangle to avoid double counting and self-interaction
        mask = torch.triu(torch.ones((n, n), dtype=torch.bool, device=device), diagonal=1)
        
        return torch.sum(total_energy_matrix[mask])

    # The harmonic_bond_energy and angle_energy methods are correct and remain the same.
    def harmonic_bond_energy(self, coords):
        if self.bond_indices.shape[0] == 0: return torch.tensor(0.0, device=coords.device, dtype=coords.dtype)
        bond_indices = self.bond_indices.to(coords.device); r0 = self.r0_tensor.to(coords.device); k = self.k_bond_tensor.to(coords.device)
        pos1, pos2 = coords[bond_indices[:, 0]], coords[bond_indices[:, 1]]
        dist = torch.norm(pos1 - pos2, dim=1)
        return (0.5 * k * (dist - r0) ** 2).sum()

    # def angle_energy(self, coords):
    #     if self.angle_indices.shape[0] == 0: return torch.tensor(0.0, device=coords.device, dtype=coords.dtype)
    #     idx, params = self.angle_indices.to(coords.device), self.angle_params.to(coords.device)
    #     A, B, C = coords[idx[:, 0]], coords[idx[:, 1]], coords[idx[:, 2]]
    #     BA, BC = A - B, C - B
    #     BA_norm, BC_norm = torch.nn.functional.normalize(BA, dim=1), torch.nn.functional.normalize(BC, dim=1)
    #     cos_angle = torch.sum(BA_norm * BC_norm, dim=1).clamp(-1.0, 1.0)
    #     angle, theta0, k_angle = torch.acos(cos_angle), params[:, 0], params[:, 1]
    #     return (0.5 * k_angle * (angle - theta0) ** 2).sum()
    
    def angle_energy(self, coords):
        """
        Calculates the harmonic angle energy in a numerically stable way,
        operating on the cosine of the angle to avoid acos and its unstable gradient.
        """
        if self.angle_indices.shape[0] == 0:
            return torch.tensor(0.0, device=coords.device, dtype=coords.dtype)
        
        idx = self.angle_indices.to(coords.device)
        params = self.angle_params.to(coords.device)
        
        # Get atom positions for each angle
        A = coords[idx[:, 0]]
        B = coords[idx[:, 1]]
        C = coords[idx[:, 2]]
        
        # Create vectors from the central atom B
        BA = A - B
        BC = C - B
        
        # Normalize the vectors to get unit vectors
        # Add a small epsilon to the norm to prevent division by zero if atoms overlap
        BA_norm = BA / (torch.norm(BA, dim=1, keepdim=True) + 1e-8)
        BC_norm = BC / (torch.norm(BC, dim=1, keepdim=True) + 1e-8)
        
        # The dot product of the unit vectors is the cosine of the angle
        # Clamping is still a good safety measure for floating point inaccuracies
        cos_theta_pred = torch.sum(BA_norm * BC_norm, dim=1).clamp(-1.0, 1.0)
        
        # Get the target angle (theta0) and force constant (k) from parameters
        theta0_rad = params[:, 0]
        k_angle = params[:, 1]
        
        # Calculate the cosine of the target angle. This is our new target value.
        cos_theta0_target = torch.cos(theta0_rad)
        
        # The loss is now the squared difference between the predicted and target cosines.
        # This is a harmonic potential in cosine space, which is stable.
        # The 0.5 * k part is kept for consistency with the harmonic potential form.
        energy = 0.5 * k_angle * (cos_theta_pred - cos_theta0_target) ** 2
        
        return energy.sum()

    def openMM_nonbonded_energy(self, coords_tensor):
        """Calculates ONLY the non-bonded energy using OpenMM force groups. Assumes coords are in nm."""
        self.context.setPositions(coords_tensor.detach().cpu().numpy() * unit.nanometer)
        state = self.context.getState(getEnergy=True, groups={1})
        return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    
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


    def __call__(self, coords_tensor, use_bonded=True, use_angle=True, use_nonbonded=True):
        E_bond = self.harmonic_bond_energy(coords_tensor) if use_bonded else torch.tensor(0.0, device=coords_tensor.device)
        E_angle = self.angle_energy(coords_tensor) if use_angle else torch.tensor(0.0, device=coords_tensor.device)
        E_nonbonded = self.nonbonded_energy(coords_tensor) if use_nonbonded else torch.tensor(0.0, device=coords_tensor.device)
        return E_bond, E_angle, E_nonbonded

# The physics_loss function remains valid and can be used as is.
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
            bond_energy, angle_energy, lj_energy = energy_calculator(coords, use_bonded=use_bonded, use_angle=use_angle, use_nonbonded=use_lj)

            # Weight the components
            # physics_energy = (
            #     5.0 * bond_energy + 
            #     2.0 * angle_energy + 
            #     0.01 * lj_energy  
            # )
            physics_energy = (
                1.0 * bond_energy + 
                1.0 * angle_energy + 
                1.0 * lj_energy  
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

# --- Example usage and verification ---
if __name__ == '__main__':
    try:
        pdb_file = 'molecule.pdb' # IMPORTANT: Replace with your actual PDB file name
        energy_calc = EnergyCalculator(pdb_file, add_hydrogens=True)
        
        initial_pos_nm = energy_calc.positions.value_in_unit(unit.nanometer)
        coords_tensor = torch.tensor(initial_pos_nm, dtype=torch.float32)
        
        # --- VERIFICATION ---
        pytorch_nonbonded_energy = energy_calc.nonbonded_energy(coords_tensor)
        openmm_nonbonded_energy = energy_calc.openMM_nonbonded_energy(coords_tensor)
        
        print(f"PyTorch Non-Bonded Energy: {pytorch_nonbonded_energy.item():.4f} kJ/mol")
        print(f"OpenMM Non-Bonded Energy:  {openmm_nonbonded_energy:.4f} kJ/mol")
        
        # Calculate difference
        diff = abs(pytorch_nonbonded_energy.item() - openmm_nonbonded_energy)
        print(f"Difference: {diff:.4f} kJ/mol")

    except Exception as e:
        print(f"An error occurred. Make sure 'molecule.pdb' exists and is a valid PDB file.")
        print(f"Error details: {e}")