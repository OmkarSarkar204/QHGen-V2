import os
import sys
import numpy as np
from tqdm import tqdm

# Add paths...
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from core.optimizer.genetic_algo_engine import QHGenOptimizer
from core.builder.slab_generator import QHSlabGenerator
from models.surrogate.schnet_engine import QHSurrogateModel
from models.quantum.azure_wrapper import QuantumValidator
from core.data.feedback_loop import ActiveLearningManager
from core.data.logger import ResearchLogger # <--- NEW

# Initialize Logger globally or pass it in (Global for simplicity here)
LAB_NOTEBOOK = None

def run_screening_cycle(generation_id=1, allowed_elements=None, logger=None):
    print(f"\n--- 🚀 STARTING QHGEN CYCLE: GENERATION {generation_id} ---")
    
    # 1. Init Tools
    optimizer = QHGenOptimizer()
    builder = QHSlabGenerator()
    # Quantum Engine (Use Azure if available)
    quantum_engine = QuantumValidator(backend="local_simulator") 
    memory = ActiveLearningManager()
    
    # Load AI with Uncertainty Capability
    model_path = "qhgen-v3/models/surrogate/weights/schnet_active_v1.pt"
    surrogate = QHSurrogateModel(loaded_weights_path=model_path if os.path.exists(model_path) else None)

    # 2. Generate Population
    population = optimizer.initialize_population(allowed_elements=allowed_elements)
    
    candidates = []
    print(">> Constructing 3D Supercells...")
    for i, comp in enumerate(tqdm(population)):
        try:
            slab = builder.build_alloy_slab(comp)
            # Add H
            z = slab.positions[:, 2]
            builder.add_hydrogen_intermediate(slab)
            
            candidates.append({
                "id": f"gen{generation_id}_cand{i}",
                "composition": comp,
                "structure": slab
            })
        except Exception:
            continue

    if not candidates: return None

    # 3. AI Scoring with UNCERTAINTY (The Fix!)
    print(">> AI Scoring with Monte Carlo Uncertainty...")
    structures = [c['structure'] for c in candidates]
    
    # Run MC Dropout
    means, stds = surrogate.predict_with_uncertainty(structures, num_samples=5)
    
    for i, cand in enumerate(candidates):
        cand['predicted_energy'] = float(means[i])
        cand['uncertainty'] = float(stds[i])
        
        # Use new exploration-heavy fitness
        cand['fitness'] = optimizer.calculate_fitness_with_exploration(
            cand['predicted_energy'], 
            cand['uncertainty'],
            exploration_weight=0.2 # Tune this: Higher = More exploration
        )

    # 4. Save COMPLETE dataset to Lab Notebook
    if logger:
        logger.log_generation(generation_id, candidates)

    # 5. Select Winner (Best Fitness)
    candidates.sort(key=lambda x: x['fitness'], reverse=True)
    winner = candidates[0]
    
    print(f"\n🏆 TOP CANDIDATE: {winner['composition']}")
    print(f"   AI Energy: {winner['predicted_energy']:.4f} eV")
    print(f"   Uncertainty: ±{winner['uncertainty']:.4f} eV") # <--- Visualizing Confidence
    
    # 6. Quantum Truth
    from core.quantum.cluster_extractor import ActiveSiteExtractor
    extractor = ActiveSiteExtractor()
    cluster = extractor.extract_cluster(winner['structure'])
    
    result = quantum_engine.validate_candidate(cluster)
    if result.get('status') == 'FAILED': return None

    true_energy = result['vqe_energy']
    print(f"   Quantum Truth: {true_energy:.4f} eV")
    
    if logger:
        logger.log_quantum_result(generation_id, winner['id'], true_energy)

    # 7. Active Learning Save
    memory.save_verified_candidate(winner['structure'], true_energy, winner['id'])
    memory.close()
    
    return {
        "composition": winner['composition'],
        "predicted_energy": winner['predicted_energy'],
        "quantum_energy": true_energy,
        "uncertainty": winner['uncertainty']
    }