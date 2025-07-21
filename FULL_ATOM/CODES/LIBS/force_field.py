import torch
import openmm as mm
from openmm import app, unit
import numpy as np

class EnergyCalculator:
    def __init__(self, pdb_file, forcefield_files=['amber99sb.xml', 'tip3p.xml'],    
                 add_hydrogens=False, prefer_gpu=True):
        """
        Initializes the OpenMM system from a PDB file and force fields.
        """
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
        
        # We don't need a full-blown integrator for single-point energy
        self.integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        
        # --- START OF REFACTORED PLATFORM AND CONTEXT INITIALIZATION ---
        
        self.context = None
        self.platform = None
        
        # Define platform configurations to try in order of preference
        platforms_to_try = []
        if prefer_gpu:
            platforms_to_try.extend([
                ('CUDA', {'CudaPrecision': 'mixed'}),
                ('CUDA', {'CudaPrecision': 'single'}),
                ('OpenCL', {'OpenCLPrecision': 'mixed'}),
                ('OpenCL', {'OpenCLPrecision': 'single'}),
            ])
        # Always add CPU as the final fallback
        platforms_to_try.append(('CPU', {}))

        # Loop through the configurations and try to create a context
        for platform_name, properties in platforms_to_try:
            try:
                platform = mm.Platform.getPlatformByName(platform_name)
                # Attempt to create the final context directly.
                # If this fails, the exception will be caught and we'll try the next config.
                self.context = mm.Context(self.system, self.integrator, platform, properties)
                self.platform = platform
                # If we succeed, break the loop
                print(f"Successfully created OpenMM context on: {platform.getName()} with properties: {properties}")
                break
            except Exception as e:
                print(f"Failed to create context on {platform_name} with {properties}. Reason: {e}")
                self.context = None # Ensure context is None if creation failed
        
        # If after all attempts, context is still None, something is seriously wrong.
        if self.context is None:
            raise RuntimeError("Could not create an OpenMM context on any available platform.")
        
        # --- END OF REFACTORED LOGIC ---
        
        # Print additional info for GPU platforms
        if self.platform.getName() in ['CUDA', 'OpenCL']:
            self._print_gpu_info()

    def __del__(self):
        """
        Explicitly clean up the OpenMM context to prevent segfaults on exit.
        """
        # The 'hasattr' check is important in case initialization failed
        if hasattr(self, 'context') and self.context is not None:
            # We must delete the context first, then the integrator
            # to release platform resources cleanly.
            # In modern OpenMM, just unsetting the context reference is often enough.
            del self.context
            del self.integrator
            # print("Cleaned up OpenMM context.") # Uncomment for debugging


    def _print_gpu_info(self):
        """Print GPU information for debugging."""
        platform_name = self.platform.getName()
        print(f"--- {platform_name} Platform Info ---")
        for prop_name in self.platform.getPropertyNames():
            try:
                value = self.context.getPlatform().getPropertyValue(self.context, prop_name)
                print(f"{prop_name}: {value}")
            except Exception:
                # Some properties might not be queryable
                pass
        print("--------------------------")

    @torch.no_grad()
    def __call__(self, coords_tensor):
        """
        Calculates the potential energy for a given set of coordinates.
        
        Args:
            coords_tensor (torch.Tensor): A tensor of shape (num_atoms, 3).
                                          Assumes coordinates are in Angstroms.
        
        Returns:
            torch.Tensor: A scalar tensor containing the potential energy in kJ/mol.
        """
        # Ensure the tensor is on the CPU and is a numpy array for OpenMM
        coords_numpy = coords_tensor.detach().cpu().numpy()
        positions = coords_numpy * unit.angstrom
        
        self.context.setPositions(positions)
        state = self.context.getState(getEnergy=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        
        return torch.tensor(energy, dtype=torch.float32)

# import torch
# import openmm as mm
# from openmm import app, unit
# import numpy as np

# class EnergyCalculator:
#     def __init__(self, pdb_file, forcefield_files=['amber99sb.xml', 'tip3p.xml'],    
#                  add_hydrogens=False):
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
        
#         self.context = mm.Context(self.system, self.integrator, 'OpenCL', {'OpenCLPrecision': 'mixed'})
#         self.platform = mm.Platform.getPlatformByName('OpenCL')
        
#         # Define platform configurations to try in order of preference
#         # platforms_to_try = []
#         # if prefer_gpu:
#         #     platforms_to_try.extend([
#         #         ('CUDA', {'CudaPrecision': 'mixed'}),
#         #         ('CUDA', {'CudaPrecision': 'single'}),
#         #         ('OpenCL', {'OpenCLPrecision': 'mixed'}),
#         #         ('OpenCL', {'OpenCLPrecision': 'single'}),
#         #     ])
#         # # Always add CPU as the final fallback
#         # platforms_to_try.append(('CPU', {}))

#         # # Loop through the configurations and try to create a context
#         # for platform_name, properties in platforms_to_try:
#         #     try:
#         #         platform = mm.Platform.getPlatformByName(platform_name)
#         #         # Attempt to create the final context directly.
#         #         # If this fails, the exception will be caught and we'll try the next config.
#         #         self.context = mm.Context(self.system, self.integrator, platform, properties)
#         #         self.platform = platform
#         #         # If we succeed, break the loop
#         #         print(f"Successfully created OpenMM context on: {platform.getName()} with properties: {properties}")
#         #         break
#         #     except Exception as e:
#         #         print(f"Failed to create context on {platform_name} with {properties}. Reason: {e}")
#         #         self.context = None # Ensure context is None if creation failed
        
#         # # If after all attempts, context is still None, something is seriously wrong.
#         # if self.context is None:
#         #     raise RuntimeError("Could not create an OpenMM context on any available platform.")
        
#         # # --- END OF REFACTORED LOGIC ---
        
#         # # Print additional info for GPU platforms
#         # if self.platform.getName() in ['CUDA', 'OpenCL']:
#         #     self._print_gpu_info()

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
#     def __call__(self, coords_tensor):
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