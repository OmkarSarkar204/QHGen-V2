import numpy as np
from ase.neighborlist import neighbor_list

class ActiveSiteExtractor:
    def __init__(self, cutoff_radius=3.5):
        # 3.5 Angstroms captures the direct bonding environment
        self.cutoff = cutoff_radius

    def extract_cluster(self, slab_atoms):
        """
        Identify the Hydrogen adsorbate and 'snip' out its nearest neighbors.
        """
        # 1. Find the Hydrogen Atom (Atomic Number 1)
        atomic_numbers = slab_atoms.get_atomic_numbers()
        h_indices = [i for i, z in enumerate(atomic_numbers) if z == 1]
        
        if not h_indices:
            # Fallback: If no H is found, return the center of the surface
            print("Warning: No Hydrogen found. Extracting surface center.")
            return self._extract_surface_center(slab_atoms)
        
        # Take the first H (usually only one in our screening)
        h_index = h_indices[0]
        
        # 2. Find Neighbors using ASE's efficient neighbor list
        # 'i' and 'j' are indices, 'd' is distance
        i_list, j_list, d_list = neighbor_list('ijd', slab_atoms, self.cutoff)
        
        # Filter for atoms connected to our Hydrogen
        neighbors = set()
        neighbors.add(h_index) # Include H itself
        
        for (i, j) in zip(i_list, j_list):
            if i == h_index:
                neighbors.add(j)
            elif j == h_index:
                neighbors.add(i)
                
        # 3. Create the Cluster Object
        # We slice the original atoms object to keep only relevant atoms
        cluster = slab_atoms[list(neighbors)]
        
        # Center in a vacuum box for clean simulation
        cluster.center(vacuum=10.0)
        
        return cluster

    def _extract_surface_center(self, slab):
        # Fallback logic if H is missing
        # Gets the highest atom (surface) and its neighbors
        z_positions = slab.positions[:, 2]
        top_atom_idx = np.argmax(z_positions)
        
        i_list, j_list, _ = neighbor_list('ijd', slab, self.cutoff)
        neighbors = {top_atom_idx}
        for (i, j) in zip(i_list, j_list):
            if i == top_atom_idx: neighbors.add(j)
            
        cluster = slab[list(neighbors)]
        cluster.center(vacuum=10.0)
        return cluster