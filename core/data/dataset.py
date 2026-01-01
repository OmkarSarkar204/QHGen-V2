import lmdb
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data

class OC20LmdbDataset(Dataset):
    """
    Robust LMDB loader for OC20 IS2RE.
    Features:
    1. Windows-Safe (Lazy Init)
    2. Legacy-Safe (Reads old PyG objects via __dict__)
    3. Warning-Free (Proper tensor conversion)
    """
    def __init__(self, lmdb_path):
        super(OC20LmdbDataset, self).__init__()
        self.path = lmdb_path
        self.env = None 
        
        # 1. Temporary open to get length
        try:
            temp_env = lmdb.open(
                self.path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False
            )
        except (lmdb.Error, IsADirectoryError):
            temp_env = lmdb.open(
                self.path, subdir=True, readonly=True, lock=False, readahead=False, meminit=False
            )

        with temp_env.begin(write=False) as txn:
            len_bytes = txn.get("length".encode("ascii"))
            if len_bytes is not None:
                self.length = pickle.loads(len_bytes)
            else:
                self.length = txn.stat()['entries']
        
        temp_env.close()

    def _connect(self):
        try:
            self.env = lmdb.open(
                self.path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False
            )
        except (lmdb.Error, IsADirectoryError):
             self.env = lmdb.open(
                self.path, subdir=True, readonly=True, lock=False, readahead=False, meminit=False
            )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.env is None:
            self._connect()

        with self.env.begin(write=False) as txn:
            key = f"{idx}".encode("ascii")
            byteflow = txn.get(key)
            if byteflow is None: return None
            
            # Load the raw legacy object
            legacy_data = pickle.loads(byteflow)

        # 2. SURGICAL EXTRACTION (Bypass PyG Version Check)
        # We access __dict__ directly to avoid triggering the "Old Version" error
        # that happens when accessing attributes like .pos on legacy objects.
        
        raw_dict = legacy_data.__dict__

        # Extract Atomic Numbers (z)
        if 'atomic_numbers' in raw_dict:
            z_raw = raw_dict['atomic_numbers']
        elif 'z' in raw_dict:
            z_raw = raw_dict['z']
        else:
            raise ValueError(f"Entry {idx} missing atomic numbers")

        # Extract Positions (pos)
        if 'pos' in raw_dict:
            pos_raw = raw_dict['pos']
        else:
            raise ValueError(f"Entry {idx} missing positions")

        # Extract Energy (y)
        if 'y_relaxed' in raw_dict:
            y_raw = raw_dict['y_relaxed']
        elif 'y' in raw_dict:
            y_raw = raw_dict['y']
        else:
            y_raw = 0.0

        # Extract System ID (sid)
        sid = raw_dict.get('sid', idx)

        # 3. CLEAN TENSOR CONSTRUCTION (Fixes UserWarnings)
        # clone().detach() is the safe way to copy data in PyTorch
        if torch.is_tensor(z_raw):
            z = z_raw.clone().detach().long()
        else:
            z = torch.as_tensor(z_raw, dtype=torch.long)

        if torch.is_tensor(pos_raw):
            pos = pos_raw.clone().detach().float()
        else:
            pos = torch.as_tensor(pos_raw, dtype=torch.float)

        if torch.is_tensor(y_raw):
            y = y_raw.clone().detach().float().view(1) # Ensure shape [1]
        else:
            y = torch.tensor([y_raw], dtype=torch.float)

        # Build FRESH Data object compatible with current PyG
        out_data = Data(z=z, pos=pos, y=y, sid=sid)
        
        return out_data