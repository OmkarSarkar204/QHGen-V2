import numpy as np
import yaml
from ase.build import fcc111, add_adsorbate
from ase.constraints import FixAtoms
from ase.visualize import view

class QHSlabGenerator:
    def __init__(self, config_path="qhgen-v3/config_alloys.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.allowed_elements = set(self.config['active_metals'])
        self.blacklist = set(self.config.get('blacklisted_elements', []))

    def validate_composition(self, elements):
        """Ensures no toxic/radioactive elements are used."""
        for el in elements:
            if el in self.blacklist:
                raise ValueError(f"CRITICAL SAFETY VIOLATION: {el} is blacklisted.")
            if el not in self.allowed_elements:
                print(f"WARNING: {el} is not in the approved active_metals list.")

    def build_alloy_slab(self, composition, size=(4, 4, 4), vacuum=10.0):
        """
        Generates a random alloy slab.
        
        Args:
            composition (dict): e.g., {'Ni': 0.8, 'Fe': 0.2}
            size (tuple): (x, y, z) unit cell repetitions. (4,4,4) approx 64 atoms.
            vacuum (float): Angstroms of vacuum padding (critical for surface physics).
            
        Returns:
            ase.Atoms: The generated slab object.
        """
        elements = list(composition.keys())
        ratios = list(composition.values())
        
        # 1. Validation
        if not np.isclose(sum(ratios), 1.0):
            raise ValueError(f"Composition ratios must sum to 1.0 (Got {sum(ratios)})")
        self.validate_composition(elements)

        # 2. Base Lattice Construction
        # We use the major element as the lattice template to determine bond distances
        major_element = elements[np.argmax(ratios)]
        
        # Create FCC (111) surface - most common catalytic face
        # size=(4,4,4) creates a slab 4 atoms wide, 4 deep, 4 layers thick
        slab = fcc111(major_element, size=size, vacuum=vacuum)
        
        # 3. Random Substitution (The "Alloy" Step)
        total_atoms = len(slab)
        atomic_numbers = slab.get_atomic_numbers()
        
        # Generate target counts for each element
        counts = {el: int(ratio * total_atoms) for el, ratio in composition.items()}
        
        # Fix rounding errors (ensure sum equals total_atoms)
        current_sum = sum(counts.values())
        diff = total_atoms - current_sum
        if diff != 0:
            # Add/subtract remainder from major element
            counts[major_element] += diff

        # Create the new list of chemical symbols
        new_symbols = []
        for el, count in counts.items():
            new_symbols.extend([el] * count)
            
        # Shuffle to simulate random alloy distribution (High Entropy effect)
        np.random.shuffle(new_symbols)
        
        # Apply new symbols to the slab
        slab.set_chemical_symbols(new_symbols)
        
        # 4. Mechanical Constraints
        # Fix the bottom 2 layers to simulate bulk material (they shouldn't move during relaxation)
        # In a 4-layer slab, we fix indices corresponding to bottom half
        # ASE atoms are often ordered by layer, but we double check Z-heights
        z_positions = slab.positions[:, 2]
        min_z = np.min(z_positions)
        # Fix atoms within 4 Angstroms of the bottom
        constraint_indices = [i for i, z in enumerate(z_positions) if z < min_z + 4.0]
        slab.set_constraint(FixAtoms(indices=constraint_indices))

        return slab

    def add_hydrogen_intermediate(self, slab):
        """Places a Hydrogen atom on the active site (Atop position)."""
        # H is placed 1.5 Angstroms above a surface atom
        z = slab.positions[:, 2]
        top_atom = np.argmax(z)
        add_adsorbate(slab, 'H', 1.5, position=slab.positions[top_atom][:2])
        return slab

# Usage Test
if __name__ == "__main__":
    gen = QHSlabGenerator()
    try:
        slab = gen.build_alloy_slab({'Ni': 0.8, 'Fe': 0.2})
        slab_with_h = gen.add_hydrogen_intermediate(slab)
        print(f"SUCCESS: Generated {slab.get_chemical_formula()} Slab")
        print(f"Total Atoms: {len(slab)}")
        # view(slab) # Uncomment to visualize window
    except Exception as e:
        print(f"FAILED: {e}")