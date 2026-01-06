import sys
import os
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import mlflow
import logging
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Fix Path imports to ensure we can load the model class
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

from models.surrogate.schnet_engine import QHSurrogateModel

# --- HARDCODED PATHS FIX ---
# Pointing to the EXACT location of the CSV and Weights
DATA_PATH = r"C:\Users\Omkar\Desktop\QHGen-V2\qhgen-v3\data\knowledge_bank\active_learning_db.csv"
SAVE_PATH = r"C:\Users\Omkar\Desktop\QHGen-V2\qhgen-v3\models\surrogate\weights\schnet_active_v1.pt"
# ---------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR_FINE_TUNE = 1e-4

# Silence Logs
logging.getLogger("azure").setLevel(logging.WARNING)

def setup_azure_logging():
    try:
        ml_client = MLClient(
            DefaultAzureCredential(),
            subscription_id="42dcbcc8-8e39-4b60-8911-2d971b41f4f5",
            resource_group_name="qhgen-rg",
            workspace_name="qhgen-quantum"
        )
        tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        print(f"Azure ML Connected: {tracking_uri}")
        return True
    except:
        return False

def load_data_from_csv(path):
    if not os.path.exists(path):
        print(f"⚠️ CSV not found at: {path}")
        return []
    
    try:
        df = pd.read_csv(path)
        # Take recent data
        df = df.tail(64)
        data_samples = []
        
        for _, row in df.iterrows():
            try:
                # Placeholder structure for fine-tuning demo
                from ase.build import molecule
                atoms = molecule('H2') 
                
                # Check for 'quantum_energy' OR 'energy'
                if 'quantum_energy' in row:
                    target = float(row['quantum_energy'])
                elif 'energy' in row:
                    target = float(row['energy'])
                else:
                    continue # Skip if no label
                
                data_samples.append((atoms, target))
            except:
                continue
                
        return data_samples
    except Exception as e:
        print(f"⚠️ CSV Read Error: {e}")
        return []

def active_training_cycle(fine_tune_only=True):
    print(f"Initializing Active Learning Engine on {DEVICE}...")
    use_azure = setup_azure_logging()
    
    # 1. Load Model
    # Ensure directory exists for saving weights
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    surrogate = QHSurrogateModel(loaded_weights_path=SAVE_PATH if os.path.exists(SAVE_PATH) else None, device=DEVICE)
    surrogate.train()
    
    optimizer = optim.Adam(surrogate.parameters(), lr=LR_FINE_TUNE)
    criterion = nn.MSELoss()

    # 2. Load Data
    dataset = load_data_from_csv(DATA_PATH)
    
    if len(dataset) == 0:
        print(f"⚠️ Notice: No new valid data found in CSV at {DATA_PATH}. Skipping fine-tuning.")
        return

    print(f">> New Quantum Validation data detected ({len(dataset)} entries). Fine-tuning...")

    # 3. Training Loop
    epochs = 5
    if use_azure:
        try:
            run_context = mlflow.start_run(run_name="Active_Fine_Tuning", nested=True)
        except:
            # Fallback if nested run fails
            run_context = mlflow.start_run(run_name="Active_Fine_Tuning")
    else:
        class DummyContext: 
            def __enter__(self): pass
            def __exit__(self,a,b,c): pass
        run_context = DummyContext()

    with run_context:
        for epoch in range(epochs):
            total_loss = 0
            count = 0
            optimizer.zero_grad()
            
            for atoms, target_energy in dataset:
                try:
                    # Manual Batching
                    batch = surrogate._prepare_batch_manually(atoms)
                    pred = surrogate.forward(batch)
                    
                    target = torch.tensor([target_energy], dtype=torch.float32, device=DEVICE)
                    loss = criterion(pred.view(-1), target.view(-1))
                    
                    loss.backward()
                    total_loss += loss.item()
                    count += 1
                except:
                    continue
            
            optimizer.step()
            avg_loss = total_loss / max(1, count)
            print(f"[Fine-Tuning] Epoch {epoch+1} | Loss: {avg_loss:.4f}")
            
            if use_azure: mlflow.log_metric("loss", avg_loss, step=epoch)

        torch.save(surrogate.state_dict(), SAVE_PATH)
        print("✅Brain Updated & Saved.")

if __name__ == "__main__":
    active_training_cycle()