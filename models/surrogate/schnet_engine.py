import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import SchNet
import numpy as np

class QHSurrogateModel:
    def __init__(self, device=None, loaded_weights_path=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize SchNet (Pre-configured architecture)
        self.model = SchNet(
            hidden_channels=128,
            num_filters=128,
            num_interactions=6,
            num_gaussians=50,
            cutoff=10.0
        ).to(self.device)
        
        if loaded_weights_path:
            self.load_weights(loaded_weights_path)

    def _ase_to_pyg(self, atoms):
      
        # Get atomic numbers (Z) and positions (pos)
        z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float)
        
        return Data(z=z, pos=pos)

    def predict_energy(self, structures_list):
        """Standard prediction (Fast)"""
        self.model.eval()
        data_list = [self._ase_to_pyg(s) for s in structures_list]
        
        if not data_list:
            return np.array([])
            
        batch = Batch.from_data_list(data_list).to(self.device)
        
        with torch.no_grad():
            energy = self.model(batch.z, batch.pos, batch.batch)
            
        return energy.cpu().detach().numpy().flatten()

    def predict_with_uncertainty(self, structures_list, num_samples=10):
        """
        ✅ MONTE CARLO PREDICTION (Slow but Smart)
        Runs the model N times with random neuron dropout to measure confidence.
        """
        self.model.train() # KEEP DROPOUT ACTIVE to simulate "confusion"
        
        data_list = [self._ase_to_pyg(s) for s in structures_list]
        if not data_list: return [], []
        
        batch = Batch.from_data_list(data_list).to(self.device)
        
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                out = self.model(batch.z, batch.pos, batch.batch)
                predictions.append(out.cpu().numpy().flatten())
        
        predictions = np.array(predictions) # Shape: (num_samples, num_structures)
        
        # Calculate Stats
        mean_energy = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return mean_energy, uncertainty

    def load_weights(self, path):
        # Using weights_only=False to support older pickle formats if needed
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        # print(f"✅ Loaded weights from {path}")

    def save_weights(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict()
        }, path)