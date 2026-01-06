import os
import csv
import json
import logging
from datetime import datetime

class ResearchLogger:
    def __init__(self, campaign_name="Default_Campaign"):
        self.master_db_path = r"C:\Users\Omkar\Desktop\QHGen-V2\qhgen-v3\data\knowledge_bank\active_learning_db.csv"
        lab_notebook_root = r"C:\Users\Omkar\Desktop\QHGen-V2\qhgen-v3\data\lab_notebook"
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        campaign_folder = f"{campaign_name}_{self.timestamp}"
        
        self.campaign_dir = os.path.join(lab_notebook_root, campaign_folder)
        
        os.makedirs(self.campaign_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.master_db_path), exist_ok=True)
        
        self.base_path = self.campaign_dir
        print(f"DEBUG: Logger initialized at: {self.campaign_dir}")
        print(f"DEBUG: Master DB targeted at: {self.master_db_path}")

    def log(self, message):
        print(message)
        log_file = os.path.join(self.campaign_dir, "session.log")
        
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")

    def log_generation(self, gen_id, best_candidate):
        energy = best_candidate.get('quantum_energy')
        if energy is None:
            energy = best_candidate.get('predicted_energy', 0.0)
        
        composition = best_candidate.get('composition', {})
        
        msg = f"GEN {gen_id} COMPLETE | Best: {composition} | E={energy:.4f} eV"
        self.log(msg)

    def log_quantum_result(self, gen_id, candidate_id, quantum_energy):
        msg = f"GEN {gen_id} | Candidate {candidate_id} | Quantum Truth: {quantum_energy:.4f} eV"
        self.log(msg)

    def log_candidate(self, candidate_data):
        json_path = os.path.join(self.campaign_dir, "candidates.json")
        data_to_save = candidate_data.copy()
        
        for k, v in data_to_save.items():
            if hasattr(v, 'item'):
                data_to_save[k] = v.item()
            
        with open(json_path, "a") as f:
            f.write(json.dumps(data_to_save) + "\n")

        try:
            energy = candidate_data.get('quantum_energy')
            if energy is None:
                energy = candidate_data.get('energy', 0.0)
            
            comp_str = json.dumps(candidate_data.get('composition', {}))
            unc = candidate_data.get('uncertainty', 0.0)
            c_id = candidate_data.get('id', 'unknown')
            
            with open(self.master_db_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([c_id, comp_str, energy, unc, self.timestamp])
                
            with open(self.master_db_path, 'r') as f:
                idx = sum(1 for line in f)
                
            self.log(f"[Memory] Candidate {c_id} saved to Knowledge Bank (Index {idx}).")
            
        except Exception as e:
            self.log(f"Error saving to Master DB: {e}")
