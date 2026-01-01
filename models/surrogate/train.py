import sys
import os
import mlflow
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# 1. FIX PATH FIRST
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.nn import L1Loss
from core.data.dataset import OC20LmdbDataset
from models.surrogate.schnet_engine import QHSurrogateModel

# CONFIGURATION
OC20_PATH = "qhgen-v3/data/raw/is2re_train.lmdb" 
NEW_DATA_PATH = "qhgen-v3/data/processed/new_quantum_findings.lmdb"
SAVE_PATH = "qhgen-v3/models/surrogate/weights/schnet_active_v1.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
LR_BASE = 1e-4
LR_FINE_TUNE = 1e-5

def setup_azure_logging():
    """Connects to Azure ML Workspace"""
    try:
        # ✅ UPDATED WITH YOUR ML CREDENTIALS
        ml_client = MLClient(
            DefaultAzureCredential(),
            subscription_id="42dcbcc8-8e39-4b60-8911-2d971b41f4f5",
            resource_group_name="qhgen-rg",
            workspace_name="qhgen-quantum" # Your ML workspace name
        )
        
        tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("QHGen_Discovery_Campaign")
        print(f"✅ Azure ML Connected: {tracking_uri}")
        return True
    except Exception as e:
        print(f"⚠️ Azure ML Connection Failed: {e}")
        print(">> Continuing with local logging only.")
        return False

def active_training_cycle(fine_tune_only=False):
    print(f"Initializing Active Learning Engine on {DEVICE}...")
    
    # Connect to Azure
    use_azure = setup_azure_logging()

    # 1. Load Base Data (if needed)
    if not fine_tune_only:
        if os.path.exists(OC20_PATH):
             base_dataset = OC20LmdbDataset(OC20_PATH)
             train_loader = DataLoader(base_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        else:
             train_loader = None
    
    # 2. Initialize Model
    surrogate = QHSurrogateModel(device=DEVICE)
    if os.path.exists(SAVE_PATH):
        print(f"Loading existing agent memory from {SAVE_PATH}")
        surrogate.load_weights(SAVE_PATH)
    
    optimizer = optim.Adam(surrogate.model.parameters(), lr=LR_BASE)
    criterion = L1Loss() 

    # 3. Check for New Data
    if os.path.exists(NEW_DATA_PATH):
        try:
            new_dataset = OC20LmdbDataset(NEW_DATA_PATH)
            if len(new_dataset) == 0:
                print("⚠️ Notice: Memory bank is empty. Skipping fine-tuning.")
            else:
                print(f">> New Quantum Validation data detected ({len(new_dataset)} entries). Fine-tuning...")
                combined_loader = DataLoader(new_dataset, batch_size=32, shuffle=True, num_workers=0)
                
                for g in optimizer.param_groups: g['lr'] = LR_FINE_TUNE
                
                train_loop(surrogate, combined_loader, optimizer, criterion, epochs=10, label="Fine-Tuning", use_azure=use_azure)
        except Exception as e:
            print(f"⚠️ Warning: Could not load active memory: {e}")

    # 4. Base Training
    if not fine_tune_only and train_loader:
        train_loop(surrogate, train_loader, optimizer, criterion, epochs=5, label="Base-Train", use_azure=use_azure)

def train_loop(agent, loader, optimizer, criterion, epochs, label, use_azure=False):
    agent.model.train()
    
    # Wrapper for MLFlow to handle both local and Azure cases gracefully
    if use_azure:
        run_context = mlflow.start_run(run_name=label, nested=True)
    else:
        # Dummy context manager if Azure is down
        class DummyContext:
            def __enter__(self): pass
            def __exit__(self, exc_type, exc_val, exc_tb): pass
        run_context = DummyContext()

    with run_context:
        if use_azure:
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("learning_rate", optimizer.param_groups[0]['lr'])
        
        for epoch in range(epochs):
            total_loss = 0
            batch_idx = 0
            
            for batch in loader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                pred = agent.model(batch.z, batch.pos, batch.batch)
                if pred.shape != batch.y.shape: pred = pred.view(batch.y.shape)
                loss = criterion(pred, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batch_idx += 1
            
            avg_loss = total_loss / max(1, batch_idx)
            print(f"[{label}] Epoch {epoch+1} | Loss: {avg_loss:.4f}")
            
            if use_azure:
                mlflow.log_metric("loss", avg_loss, step=epoch)
            
            agent.save_weights(SAVE_PATH)

if __name__ == "__main__":
    active_training_cycle()