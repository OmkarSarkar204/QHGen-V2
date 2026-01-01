import lmdb
import pickle
import os
import torch

# Global class for pickling
class StorageObject:
    def __init__(self, d):
        self.__dict__ = d

class ActiveLearningManager:
    def __init__(self, db_path="qhgen-v3/data/processed/new_quantum_findings.lmdb"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Windows-Safe Map Size (1GB)
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
        # 1. Convert to Data Dictionary
        data_dict = {
            "atomic_numbers": structure.get_atomic_numbers(),
            "pos": structure.get_positions(),
            "y_relaxed": float(energy),
            "sid": run_id
        }
        
        obj_to_save = StorageObject(data_dict)

        # 2. Write to LMDB
        with self.env.begin(write=True) as txn:
            # FIX: Read the exact length counter instead of guessing with stat()
            length_bytes = txn.get("length".encode("ascii"))
            
            if length_bytes:
                next_idx = pickle.loads(length_bytes)
            else:
                next_idx = 0

            # Use the strictly sequential index
            key = f"{next_idx}".encode("ascii")
            value = pickle.dumps(obj_to_save)
            
            txn.put(key, value)
            
            # Increment and save the new length
            txn.put("length".encode("ascii"), pickle.dumps(next_idx + 1))

        print(f"✅ [Memory] Candidate {run_id} saved to Active Knowledge Bank (Index {next_idx}).")

    def close(self):
        self.env.close()