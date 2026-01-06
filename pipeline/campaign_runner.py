import time
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from pipeline.screening_flow import run_screening_cycle
from models.surrogate.train import active_training_cycle
from core.agent.summary import AIchemist
from core.data.logger import ResearchLogger

def get_user_constraints():
    
    print("Which elements do you want to investigate?")
    print("Enter symbols separated by space (e.g., 'Ni Mo' or 'fe ni').")
    print("Press ENTER to let the AI decide freely.")
    
    user_input = input(">> Target Elements: ").strip()
    
    if not user_input:
        print(">> No input provided. AI will explore the full Periodic Table.")
        return None
    
    elements = user_input.split()
    valid_elements = ['Ni', 'Co', 'Fe', 'Cu', 'Mo', 'W', 'Pt', 'Pd', 'Ti', 'Au', 'Ag']
    cleaned_elements = []
    
    for e in elements:
        formatted = e.capitalize()
        if formatted in valid_elements:
            cleaned_elements.append(formatted)
        else:
            print(f"Ignoring unknown or unsupported element: {e}")
    
    if not cleaned_elements:
        print("No valid elements found. Defaulting to full search.")
        return None

    print(f"Constraining search space to: {cleaned_elements}")
    return cleaned_elements

def run_hybrid_discovery(generations=5):
    target_elements = get_user_constraints()
    
    campaign_name = "Hybrid_Campaign"
    if target_elements:
        campaign_name += "_" + "".join(target_elements)
    
    lab_notebook = ResearchLogger(campaign_name=campaign_name)
    scientist = AIchemist()
    campaign_history = []
    
    print(f"\n--- STARTING HYBRID DISCOVERY CAMPAIGN ({generations} Gens) ---")
    
    for i in range(1, generations + 1):
        print(f"\n=== GENERATION {i} ===")
        
        winner_data = run_screening_cycle(
            generation_id=i,
            allowed_elements=target_elements,
            logger=lab_notebook
        )
        
        if winner_data:
            campaign_history.append({
                "generation": i,
                "composition": winner_data['composition'],
                "energy": winner_data['quantum_energy'],
                "ai_pred": winner_data['predicted_energy'],
                "uncertainty": winner_data.get('uncertainty', 0.0)
            })
        
        print(f"\n=== TRAINING CYCLE {i} ===")
        active_training_cycle(fine_tune_only=True)
        time.sleep(1)

    print("\n==============================================")
    print("GENERATING FINAL REPORT")
    print("==============================================")
    
    if campaign_history:
        final_report = scientist.generate_campaign_summary(campaign_history)
        print(final_report)
        
        report_path = os.path.join(lab_notebook.base_path, "Final_Scientific_Report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(final_report)
        print(f"Report saved to: {report_path}")
        print(f"All data stored in: {lab_notebook.base_path}")
    else:
        print("No discoveries made.")

    print("\nCAMPAIGN FINISHED")

if __name__ == "__main__":
    run_hybrid_discovery(generations=5)
