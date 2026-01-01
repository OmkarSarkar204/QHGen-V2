import os
import sys
import numpy as np

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from core.builder.slab_generator import QHSlabGenerator
from models.surrogate.schnet_engine import QHSurrogateModel
from models.quantum.azure_wrapper import QuantumValidator

def get_user_composition():
    """
    Parses user input like "Ni 0.8 Fe 0.2" into a dictionary.
    """
    print("\n--- 🧪 QHGen V3.0 Manual Composition Entry ---")
    print("Enter your alloy formula using Element Symbol and Ratio.")
    print("Example: Ni 0.8 Fe 0.2  (Ratios must sum to 1.0)")
    
    user_input = input(">> Enter Composition: ").strip()
    
    try:
        parts = user_input.split()
        if len(parts) % 2 != 0:
            raise ValueError("Invalid format. Must be pairs of 'Element Ratio'.")
            
        composition = {}
        total_ratio = 0.0
        
        for i in range(0, len(parts), 2):
            element = parts[i]
            ratio = float(parts[i+1])
            composition[element] = ratio
            total_ratio += ratio
            
        # Normalization Check
        if not np.isclose(total_ratio, 1.0):
            print(f"⚠️  Warning: Ratios sum to {total_ratio}. Normalizing to 1.0...")
            for k in composition:
                composition[k] /= total_ratio
                
        return composition
        
    except Exception as e:
        print(f"❌ Input Error: {e}")
        return None

def test_my_alloy(composition_dict):
    print(f"\n--- 🔬 ANALYZING: {composition_dict} ---")
    
    # 1. Initialize Tools
    builder = QHSlabGenerator()
    quantum = QuantumValidator(backend="local_simulator") 
    
    # Load AI
    model_path = "qhgen-v3/models/surrogate/weights/schnet_active_v1.pt"
    if os.path.exists(model_path):
        surrogate = QHSurrogateModel(loaded_weights_path=model_path)
        has_ai = True
    else:
        has_ai = False

    # 2. Build Structure
    print(">> Building Atomic Structure...")
    try:
        slab = builder.build_alloy_slab(composition_dict)
        # Add Hydrogen
        z = slab.positions[:, 2]
        top_atom = np.argmax(z)
        builder.add_hydrogen_intermediate(slab)
    except Exception as e:
        print(f"❌ Error building slab: {e}")
        return

    # 3. AI Opinion
    if has_ai:
        raw_pred = surrogate.predict_energy([slab])
        pred_energy = float(raw_pred[0].item())
        print(f"🤖 AI Prediction: {pred_energy:.4f} eV")
    
    # 4. Quantum Validation
    from core.quantum.cluster_extractor import ActiveSiteExtractor
    extractor = ActiveSiteExtractor()
    cluster = extractor.extract_cluster(slab)
    
    print(f"⚛️  Running Quantum Validation on {len(cluster)} atoms...")
    result = quantum.validate_candidate(cluster)
    
    if result.get('status') == 'FAILED':
        print("❌ Quantum Validation Failed (Geometry Error).")
        return

    dg = result['vqe_energy']
    print(f"🎯 Quantum Truth (Delta G): {dg:.4f} eV")
    
    # 5. Final Verdict
    if abs(dg) < 0.1:
        print("✅ VERDICT: Excellent Catalyst (Sabatier Optimum)")
    elif dg < -0.3:
        print("⚠️ VERDICT: Binds too strongly (Poisoning risk)")
    else:
        print("⚠️ VERDICT: Binds too weakly (Inefficient)")

if __name__ == "__main__":
    # Loop to allow multiple tests without restarting script
    while True:
        comp = get_user_composition()
        if comp:
            test_my_alloy(comp)
        
        cont = input("\nTest another? (y/n): ").lower()
        if cont != 'y':
            break