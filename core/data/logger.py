import os
import csv
import json
import time
from datetime import datetime
from ase.io import write

class ResearchLogger:
    def __init__(self, campaign_name="experiment"):
        # Create a unique timestamped folder for this entire run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_path = f"qhgen-v3/data/lab_notebook/{campaign_name}_{timestamp}"
        os.makedirs(self.base_path, exist_ok=True)
        
        print(f"📂 Research Log Initialized: {self.base_path}")

    def log_generation(self, gen_id, population_data):
        """
        Saves all data for a specific generation.
        population_data: List of dicts containing {composition, energy, uncertainty, structure}
        """
        gen_folder = os.path.join(self.base_path, f"gen_{gen_id}")
        struct_folder = os.path.join(gen_folder, "structures")
        os.makedirs(struct_folder, exist_ok=True)

        csv_path = os.path.join(gen_folder, "data_summary.csv")
        
        # 1. Save CSV Summary
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(["ID", "Composition", "Predicted_Energy", "Uncertainty", "Fitness_Score", "Status"])
            
            for i, cand in enumerate(population_data):
                # Save structure file (XYZ) for visual analysis later
                filename = f"cand_{i}.xyz"
                write(os.path.join(struct_folder, filename), cand['structure'])
                
                # Write row
                comp_str = str(cand['composition'])
                writer.writerow([
                    i, 
                    comp_str, 
                    f"{cand.get('predicted_energy', 0):.4f}",
                    f"{cand.get('uncertainty', 0):.4f}",
                    f"{cand.get('fitness', 0):.4f}",
                    "Analyzed"
                ])
        
        # 2. Save Metadata (Hyperparameters)
        with open(os.path.join(gen_folder, "meta.json"), 'w') as f:
            json.dump({"timestamp": time.time(), "count": len(population_data)}, f)

    def log_quantum_result(self, gen_id, candidate_id, result):
        """Logs the expensive Quantum VQE result separately"""
        log_file = os.path.join(self.base_path, "quantum_validation_log.txt")
        with open(log_file, "a") as f:
            f.write(f"GEN {gen_id} | ID {candidate_id} | Truth: {result} eV\n")