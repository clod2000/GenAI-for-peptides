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

# --- Plotting Functions ---

def get_color_map(peptide_graph):
    """Creates a color map based on atom type (e.g., element)."""
    # Using the first feature (assumed atomic number or type index)
    atom_type_feature = peptide_graph.x[:, 0].numpy()
    colors = []
    # Basic color map (customize as needed)
    color_dict = {
        1: 'lightgrey',  # Hydrogen
        6: 'black',      # Carbon
        7: 'blue',       # Nitrogen
        8: 'red',        # Oxygen
        16: 'yellow',    # Sulfur
        15: 'orange',    # Phosphorus
        0: 'purple'      # Unknown/Other
    }
    for feature_val in atom_type_feature:
        colors.append(color_dict.get(int(feature_val), 'purple')) # Use int() in case feature is float
    return colors

def plot_graph_2d(peptide_graph, title="Peptide Graph (2D Projection)"):
    """Plots the graph as a 2D projection using NetworkX and Matplotlib."""
    plt.figure(figsize=(15, 10))

    # Convert PyG Data object to NetworkX graph
    # node_attrs=["x", "pos"] transfers features and positions if needed by NX, but we'll use pos separately
    # graph_attrs=["name"] could transfer graph-level attributes
    g = to_networkx(peptide_graph, node_attrs=None, edge_attrs=None, to_undirected=True)

    # Get positions (use XY coordinates for 2D projection)
    pos_2d = {i: tuple(coords[:2]) for i, coords in enumerate(peptide_graph.pos.numpy())}

    # Get node colors based on atom type
    colors = get_color_map(peptide_graph)

    # Get node labels (e.g., atom names or types) - Optional
    # labels = {i: name for i, name in enumerate(peptide_graph.atom_names)}
    #print(peptide_graph.x[:, 0].item())
    # labels = {i: f"{name}({int(peptide_graph.x[i, 0].item())})" # Name(TypeFeature)
    #           for i, name in enumerate(peptide_graph.atoms)}

    # Create labels using only atomic numbers from node features
    labels = {i: f"({int(peptide_graph.x[i, 0].item())})" for i in range(peptide_graph.num_nodes)}
    

    nx.draw_networkx_edges(g, pos_2d, alpha=0.6, edge_color='gray')
    nx.draw_networkx_nodes(g, pos_2d, node_color=colors, node_size=150) # Adjust node_size
    nx.draw_networkx_labels(g, pos_2d, labels=labels, font_size=8) # Adjust font_size

    plt.title(title)
    plt.xlabel("X coordinate (projection)")
    plt.ylabel("Y coordinate (projection)")
    plt.gca().set_aspect('equal', adjustable='box') # Try to keep aspect ratio
    plt.xticks([]) # Hide axis ticks
    plt.yticks([])
    plt.box(False) # Remove frame

    plt.savefig(f"{title}_2d.png", dpi=300) # Save the figure
    plt.show()


def plot_graph_3d(peptide_graph, title="Peptide Graph (3D)"):
    """Plots the graph in 3D using Matplotlib."""
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    pos_3d = peptide_graph.pos.numpy()
    edge_index = peptide_graph.edge_index.numpy()
    colors = get_color_map(peptide_graph)
   # atom_names = peptide_graph.atom_names

    # Plot nodes
    ax.scatter(pos_3d[:, 0], pos_3d[:, 1], pos_3d[:, 2], c=colors, s=100, edgecolors='k', alpha=0.8) # s=size

    # Plot edges
    if edge_index.shape[1] > 0: # Check if there are edges
        for i in range(edge_index.shape[1]):
            idx1 = edge_index[0, i]
            idx2 = edge_index[1, i]
            # Draw line between the two points
            ax.plot([pos_3d[idx1, 0], pos_3d[idx2, 0]],
                    [pos_3d[idx1, 1], pos_3d[idx2, 1]],
                    [pos_3d[idx1, 2], pos_3d[idx2, 2]],
                    'k-', alpha=0.5, linewidth=1.0) # Black line for edges

    # Optional: Add labels (can get crowded in 3D)
    # for i, name in enumerate(atom_names):
    #      ax.text(pos_3d[i, 0], pos_3d[i, 1], pos_3d[i, 2], f"{name}", size=8, zorder=1, color='k')


    ax.set_xlabel("X Coordinate (Å)")
    ax.set_ylabel("Y Coordinate (Å)")
    ax.set_zlabel("Z Coordinate (Å)")
    ax.set_title(title)

    # Try to make aspect ratio equal - may require manual adjustment
    # Getting true 'equal' aspect in Matplotlib 3D can be tricky
    xyz_limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).T
    max_range = np.diff(xyz_limits).max() * 2.0
    mean_x = np.mean(xyz_limits[:, 0])
    mean_y = np.mean(xyz_limits[:, 1])
    mean_z = np.mean(xyz_limits[:, 2])
    ax.set_xlim3d(mean_x - max_range, mean_x + max_range)
    ax.set_ylim3d(mean_y - max_range, mean_y + max_range)
    ax.set_zlim3d(mean_z - max_range, mean_z + max_range)
    # Alternative way using box aspect (newer Matplotlib versions)
    # ax.set_box_aspect([1,1,1]) # Ratio of axis lengths

    plt.savefig(f"{title}.png", dpi=300) # Save the figure

    plt.show()




# --- Configuration & Helper Functions ---

# Suppress warnings (optional)
warnings.filterwarnings('ignore', category=UserWarning, module='MDAnalysis')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Feature engineering function 
def get_atom_feature(atom):
    features = []
    atomic_num_map = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16, 'P': 15}

    # 1. Feature based on Atom Type (from Topology)
    atom_type_feature = 0
    if hasattr(atom, 'type') and atom.type:
        atom_type = atom.type.upper()
        # --- BEGIN CUSTOMIZATION ---
        # Basic heuristic (MUST BE ADAPTED FOR YOUR FORCE FIELD)
        type_prefix = ''.join(filter(str.isalpha, atom_type))
        if type_prefix.startswith('H'): atom_type_feature = 1
        elif type_prefix.startswith('C'): atom_type_feature = 6
        elif type_prefix.startswith('N'): atom_type_feature = 7
        elif type_prefix.startswith('O'): atom_type_feature = 8
        elif type_prefix.startswith('S'): atom_type_feature = 16
        elif type_prefix.startswith('P'): atom_type_feature = 15
        else:
             try: atom_type_feature = atomic_num_map.get(atom.element.upper(), 0)
             except (AttributeError, TypeError): atom_type_feature = 0
        # --- END CUSTOMIZATION ---
    else: # Fallback
        try: atom_type_feature = atomic_num_map.get(atom.element.upper(), 0)
        except (AttributeError, TypeError): atom_type_feature = 0
    features.append(atom_type_feature)

    # 2. Partial Charge (from Topology)
    charge = 0.0
    if hasattr(atom, 'charge'):
        try: charge = float(atom.charge)
        except (ValueError, TypeError): charge = 0.0
    features.append(charge)

    # Add more static features if desired (e.g., mass)
    if hasattr(atom, 'mass'): features.append(atom.mass)

    return features



# --- PyTorch Geometric Dataset Class ---

class TrajectoryDataset(InMemoryDataset): # Inherit from InMemoryDataset
    def __init__(self, root, tpr_filename, trajectory_filename, selection='protein', transform=None, pre_transform=None, pre_filter=None):
        self.tpr_filename = tpr_filename
        self.trajectory_filename = trajectory_filename
        self.selection = selection
        # Call InMemoryDataset's __init__
        super().__init__(root, transform, pre_transform, pre_filter)
        # InMemoryDataset loads data automatically using self.processed_paths
        #self.data, self.slices = torch.load(self.processed_paths[0])

        try:
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
            print(f"Successfully loaded processed data from {self.processed_paths[0]}")
        except FileNotFoundError:
            print(f"Processed file not found at {self.processed_paths[0]}. Will run process().")
            # If the file doesn't exist, __init__ will finish, and the PyG framework
            # should call _process() which calls process() to create the file.
        except Exception as e:
            print(f"Error loading processed file {self.processed_paths[0]}: {e}")
            # Re-raise or handle appropriately if loading fails for other reasons
            raise e


    # raw_dir, processed_dir properties are inherited

    @property
    def raw_file_names(self):
        return [self.tpr_filename, self.trajectory_filename]

    @property
    def processed_file_names(self):
        tpr_base = osp.splitext(self.tpr_filename)[0]
        sel_str = self.selection.replace(' ', '_').replace('(', '').replace(')', '')
        return [f'data_{tpr_base}_{sel_str}.pt'] # Should return a list

    def download(self):
        # ... (download/check logic remains the same) ...
        raw_tpr_path = osp.join(self.raw_dir, self.tpr_filename)
        raw_traj_path = osp.join(self.raw_dir, self.trajectory_filename)
        if not osp.exists(raw_tpr_path) or not osp.exists(raw_traj_path):
             raise FileNotFoundError(f"Raw files not found! Please place "
                                     f"'{self.tpr_filename}' and "
                                     f"'{self.trajectory_filename}' in {self.raw_dir}")

    def process(self):
        # --- Load Universe ---
        tpr_path = osp.join(self.raw_dir, self.tpr_filename)
        traj_path = osp.join(self.raw_dir, self.trajectory_filename)
        print(f"Loading Universe: TPR='{tpr_path}', Trajectory='{traj_path}'")
        try:
            u = mda.Universe(tpr_path, traj_path)
        except Exception as e:
            print(f"Error loading MDAnalysis Universe: {e}")
            return

        # --- Select Atoms ---
        print(f"Selecting atoms with: '{self.selection}'")
        atoms = u.select_atoms(self.selection)
        n_atoms = len(atoms)
        if n_atoms == 0:
            print(f"Warning: Selection '{self.selection}' resulted in 0 atoms. Aborting.")
            return
        print(f"Selected {n_atoms} atoms.")

        # --- Pre-calculate static information ---
        print("Calculating static features and topology...")
        static_features = []
        atom_names = []
        atom_types = []
        resnames = []
        resids = []
        atom_indices_map = {atom.index: i for i, atom in enumerate(atoms)} # For bond mapping

        for atom in atoms:
             static_features.append(get_atom_feature(atom))
             atom_names.append(atom.name)
             atom_types.append(getattr(atom, 'type', getattr(atom, 'name', 'UNK')))
             resnames.append(getattr(atom, 'resname', 'UNK'))
             resids.append(getattr(atom, 'resid', -1))
        x_static = torch.tensor(static_features, dtype=torch.float)

        # Edge Index
        edge_list = []
        if hasattr(atoms, 'bonds') and len(atoms.bonds) > 0:
            print(f"Found {len(atoms.bonds)} bonds in topology for the selection.")
            for bond in atoms.bonds:
                idx1_global = bond.atoms[0].index
                idx2_global = bond.atoms[1].index
                local_idx1 = atom_indices_map.get(idx1_global)
                local_idx2 = atom_indices_map.get(idx2_global)
                if local_idx1 is not None and local_idx2 is not None:
                    edge_list.append([local_idx1, local_idx2])
                    edge_list.append([local_idx2, local_idx1])
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                print(f"Created edge_index with shape: {edge_index.shape}")
            else: edge_index = torch.empty((2, 0), dtype=torch.long)
        else: edge_index = torch.empty((2, 0), dtype=torch.long)

        # --- Process trajectory frames ---
        data_list = []
        num_frames = len(u.trajectory)
        print(f"Processing {num_frames} frames...")
        for i, ts in enumerate(u.trajectory):
            if (i + 1) % 100 == 0: print(f"  Frame {i+1}/{num_frames} (Time: {ts.time:.2f} ps)")

            pos = torch.tensor(atoms.positions, dtype=torch.float)
            data = Data(x=x_static.clone(), edge_index=edge_index.clone(), pos=pos)

            # Add other static info (lists are less efficient for InMemoryDataset, consider tensorizing if needed)
            # data.atom_names = atom_names # Storing lists per graph can be inefficient
            # data.atom_types = atom_types
            # data.resnames = resnames
            # data.resids = resids
            # Store single values:
            data.frame_index = torch.tensor([i], dtype=torch.long) # Store as tensor for collation
            data.time = torch.tensor([ts.time], dtype=torch.float) # Store as tensor

            # Apply pre_filter and pre_transform (as defined for InMemoryDataset)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        # --- Save the Processed Data (InMemoryDataset handles collation) ---
        if not data_list:
            print("Warning: No data objects generated. Saving empty dataset.")
            data, slices = self.collate([]) # Collate empty list
        else:
            print(f"Collating {len(data_list)} graphs...")
            data, slices = self.collate(data_list) # Use the inherited collate method

        print(f"Saving collated data and slices to {self.processed_paths[0]}...")
        torch.save((data, slices), self.processed_paths[0])
        print(f"Processing complete.")

   
  
















# --- Example Usage ---
if __name__ == '__main__':
    ROOT_DIR = '.'
    # MODIFY these filenames
    TPR_FILE = 'MD.tpr'       # Place generated TPR in ./raw/
    TRAJECTORY = 'MD_with_solvent_noPBC.xtc' # Place in ./raw/
    SELECTION = 'protein'

    os.makedirs(osp.join(ROOT_DIR, 'raw'), exist_ok=True)
    os.makedirs(osp.join(ROOT_DIR, 'processed'), exist_ok=True)

    print("Creating/Loading Dataset...")
    try:
        # Update the instantiation to use tpr_filename
        dataset = TrajectoryDataset(root=ROOT_DIR,
                                    tpr_filename=TPR_FILE,
                                    trajectory_filename=TRAJECTORY,
                                    selection=SELECTION)

        # Force processing if needed (PyG usually handles this, but can force)
        # if not osp.exists(dataset.processed_paths[0]):
        #    print("Processed file not found, explicitly calling process...")
        #    dataset.process()
        #    # Need to reload data/slices after explicit process call if not done by PyG framework
        #    if osp.exists(dataset.processed_paths[0]):
        #        dataset.data, dataset.slices = torch.load(dataset.processed_paths[0])
        #    else:
        #        print("Processing failed to create the output file.")


        if len(dataset) > 0:
             print(f"\nDataset created/loaded successfully!")
             print(f"Number of graphs (frames): {len(dataset)}")
             # ... (rest of the example usage: printing graph info, plotting) ...
             first_graph = dataset[0]
             if first_graph: # Check if get returned a graph
                print("\n--- First Graph (Frame 0) ---")
                print(first_graph)
             # ... plotting etc ...
        else:
            print("\nDataset is empty or processing failed.")


    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(f"Please make sure '{TPR_FILE}' and '{TRAJECTORY}' exist in the './raw/' directory.")
        print("You may need to generate the TPR file using 'gmx grompp'.")
    except Exception as e:
        import traceback
        print(f"\nAn unexpected error occurred:")
        traceback.print_exc()


    # Optional: Plot the first graph
    if 'first_graph' in locals() and first_graph is not None:
        print("\n--- Plotting Graphs ---")

        # Plot 2D Projection
        try:
            plot_graph_2d(first_graph, title=f"Tetraalanine Graph (2D Projection - {first_graph.num_nodes} nodes, {first_graph.num_edges // 2} bonds)")
        except Exception as e:
            print(f"Could not plot 2D graph: {e}")


        # Plot 3D Representation
        try:
            plot_graph_3d(first_graph, title=f"Tetraalanine Graph (3D - {first_graph.num_nodes} nodes, {first_graph.num_edges // 2} bonds)")
        except Exception as e:
            print(f"Could not plot 3D graph: {e}")

    else:
        print("Variable 'first_graph' not found. Please run the graph creation code first.")      # Optional: Plot the first graph