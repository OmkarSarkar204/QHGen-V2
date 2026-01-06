import os
import yaml
import numpy as np
from ase.build import fcc111, add_adsorbate
from ase.constraints import FixAtoms

class QHSlabGenerator:
    def __init__(self, config_path=None):
        if config_path is None:
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_script_dir))
            self.config_path = os.path.join(project_root, "config_alloys.yaml")
        else:
            self.config_path = config_path

        print(f"DEBUG: Loading config from: {self.config_path}")

        if not os.path.exists(self.config_path):
            if os.path.exists("config_alloys.yaml"):
                self.config_path = "config_alloys.yaml"
            else:
                raise FileNotFoundError(f"Could not find config_alloys.yaml at {self.config_path}")

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.allowed_elements = set(self.config['active_metals'])
        self.blacklist = set(self.config.get('blacklisted_elements', []))

    def validate_composition(self, elements):
        for el in elements:
            if el in self.blacklist:
                raise ValueError(f"CRITICAL SAFETY VIOLATION: {el} is blacklisted.")
            if el not in self.allowed_elements:
                print(f"WARNING: {el} is not in the approved active_metals list.")

    def build_alloy_slab(self, composition, size=(4, 4, 4), vacuum=10.0):
        elements = list(composition.keys())
        ratios = list(composition.values())
        
        if not np.isclose(sum(ratios), 1.0):
            raise ValueError(f"Composition ratios must sum to 1.0 (Got {sum(ratios)})")
        self.validate_composition(elements)

        major_element = elements[np.argmax(ratios)]
        slab = fcc111(major_element, size=size, vacuum=vacuum)
        
        total_atoms = len(slab)
        atomic_numbers = slab.get_atomic_numbers()
        
        counts = {el: int(ratio * total_atoms) for el, ratio in composition.items()}
        
        current_sum = sum(counts.values())
        diff = total_atoms - current_sum
        if diff != 0:
            counts[major_element] += diff

        new_symbols = []
        for el, count in counts.items():
            new_symbols.extend([el] * count)
            
        np.random.shuffle(new_symbols)
        slab.set_chemical_symbols(new_symbols)
        
        z_positions = slab.positions[:, 2]
        min_z = np.min(z_positions)
        constraint_indices = [i for i, z in enumerate(z_positions) if z < min_z + 4.0]
        slab.set_constraint(FixAtoms(indices=constraint_indices))

        return slab

    def add_hydrogen_intermediate(self, slab):
        z = slab.positions[:, 2]
        top_atom = np.argmax(z)
        add_adsorbate(slab, 'H', 1.5, position=slab.positions[top_atom][:2])
        return slab

if __name__ == "__main__":
    gen = QHSlabGenerator()
    try:
        slab = gen.build_alloy_slab({'Ni': 0.8, 'Fe': 0.2})
        slab_with_h = gen.add_hydrogen_intermediate(slab)
        print(f"SUCCESS: Generated {slab.get_chemical_formula()} Slab")
        print(f"Total Atoms: {len(slab)}")
    except Exception as e:
        print(f"FAILED: {e}")
