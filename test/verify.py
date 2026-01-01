import torch
import torch_geometric
import ase
import lmdb
import sys
import os

def check_component(name, version_check=None):
    try:
        if name == "torch":
            print(f"✅ Torch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
        elif name == "pyg":
            print(f"✅ PyG: {torch_geometric.__version__}")
        elif name == "ase":
            print(f"✅ ASE: {ase.__version__}")
        elif name == "lmdb":
            print(f"✅ LMDB: {lmdb.__version__}")
        elif name == "data":
            # Check if data folders exist
            if os.path.exists("qhgen-v3/data/raw"):
                 print("✅ Data Folder Structure: OK")
            else:
                 print("❌ Data Folder Structure: MISSING")
    except ImportError:
        print(f"❌ {name}: NOT INSTALLED")

print("--- QHGen V3.0 Environment Verification ---")
check_component("torch")
check_component("pyg")
check_component("ase")
check_component("lmdb")
check_component("data")

# Check GPU Memory for OC20 training
if torch.cuda.is_available():
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"ℹ️  GPU Memory: {mem:.2f} GB")
    if mem < 8.0:
        print("⚠️  WARNING: <8GB VRAM. Training SchNet on OC20 might OOM. Reduce batch size.")
else:
    print("⚠️  WARNING: Running on CPU. Training will be extremely slow.")

print("-------------------------------------------")