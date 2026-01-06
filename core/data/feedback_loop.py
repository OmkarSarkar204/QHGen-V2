import lmdb
import pickle
import os
import torch
import pandas as pd
import numpy as np

class StorageObject:
    def __init__(self, d):
        self.__dict__ = d

class ActiveLearningManager:
    def __init__(self, db_path=None):
        self.csv_path = r"C:\Users\Omkar\Desktop\QHGen-V2\qhgen-v3\data\knowledge_bank\active_learning_db.csv"
        self.db_path = r"C:\Users\Omkar\Desktop\QHGen-V2\qhgen-v3\data\processed\new_quantum_findings.lmdb"

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        safe_map_size = 1024 * 1024 * 1024 
        
        self.env = lmdb.open(
            self.db_path,
            map_size=safe_map_size,
            subdir=False,
            readonly=False,
            meminit=False,
            map_async=True,
        )

    def save_verified_candidate(self, structure, energy, run_id):
        data_dict = {
            "atomic_numbers": structure.get_atomic_numbers(),
            "pos": structure.get_positions(),
            "y_relaxed": float(energy),
            "sid": run_id
        }
        obj = StorageObject(data_dict)
        with self.env.begin(write=True) as txn:
            len_bytes = txn.get("length".encode("ascii"))
            idx = pickle.loads(len_bytes) if len_bytes else 0
            txn.put(f"{idx}".encode("ascii"), pickle.dumps(obj))
            txn.put("length".encode("ascii"), pickle.dumps(idx + 1))
        print(f"[Memory] Candidate {run_id} saved to Active Knowledge Bank (Index {idx}).")

    def load_active_learning_data(self):
        if not os.path.exists(self.csv_path):
            print(f"CSV NOT FOUND at: {self.csv_path}")
            print("This is expected for the very first run. It will be created soon.")
            return None
        
        try:
            df = pd.read_csv(self.csv_path)
            
            if len(df) < 5:
                print("Not enough data points for fine-tuning yet (Need > 5).")
                return None
                
            print(f"Loaded {len(df)} verified candidates from Knowledge Bank.")
            return df
            
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return None

    def close(self):
        self.env.close()
