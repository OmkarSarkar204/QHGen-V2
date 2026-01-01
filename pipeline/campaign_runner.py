import time
import sys
import os

# --- 1. SETUP PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

# --- 2. IMPORT MODULES ---
from pipeline.screening_flow import run_screening_cycle
from models.surrogate.train import active_training_cycle
from core.agent.summary import AIchemist
from core.data.logger import ResearchLogger  # Ensure core/data/logger.py exists!

def get_user_constraints():
    """
    Asks the user for input elements and handles case-sensitivity automatically.
    """
    print("\n=============================================")
    print("   🧪 QHGen RESEARCHER MODE (Hybrid AI)   ")
    print("=============================================")
    print("Which elements do you want to investigate?")
    print("Enter symbols separated by space (e.g., 'Ni Mo' or 'fe ni').")
    print("Press ENTER to let the AI decide freely.")
    
    user_input = input(">> Target Elements: ").strip()
    
    if not user_input:
        print(">> No input provided. AI will explore the full Periodic Table.")
        return None
    
    elements = user_input.split()
    
    # --- SMART CAPITALIZATION LOGIC ---
    valid_elements = ['Ni', 'Co', 'Fe', 'Cu', 'Mo', 'W', 'Pt', 'Pd', 'Ti', 'Au', 'Ag']
    cleaned_elements = []
    
    for e in elements:
        # Convert "fe" -> "Fe", "NI" -> "Ni"
        formatted = e.capitalize()
        if formatted in valid_elements:
            cleaned_elements.append(formatted)
        else:
            print(f"⚠️  Ignoring unknown or unsupported element: {e}")
    
    if not cleaned_elements:
        print("❌ No valid elements found. Defaulting to full search.")
        return None

    print(f">> ✅ Constraining search space to: {cleaned_elements}")
    return cleaned_elements

def run_hybrid_discovery(generations=5):
    # 1. Ask User for Input
    target_elements = get_user_constraints()
    
    # 2. Initialize Scientific Tools
    campaign_name = "Hybrid_Campaign"
    if target_elements:
        campaign_name += "_" + "".join(target_elements)
    
    # The 'Lab Notebook' saves every alloy structure to a CSV/Folder
    lab_notebook = ResearchLogger(campaign_name=campaign_name)
    
    # The 'Scientist' writes the final report
    scientist = AIchemist()
    
    campaign_history = []
    
    print(f"\n--- 🌍 STARTING HYBRID DISCOVERY CAMPAIGN ({generations} Gens) ---")
    
    for i in range(1, generations + 1):
        print(f"\n\n=== 🔄 GENERATION {i} ===")
        
        # 3. RUN CYCLE (Pass Constraints + Logger)
        # The screening flow will now log uncertainty and save data to the notebook
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
                # Add uncertainty to history for the final report
                "uncertainty": winner_data.get('uncertainty', 0.0) 
            })
        
        # 4. TRAIN AI (Active Learning)
        print(f"\n=== 🧠 TRAINING CYCLE (Gen {i}) ===")
        # Fine-tune the surrogate model on the new Quantum Truth data
        active_training_cycle(fine_tune_only=True)
        
        # Brief pause to ensure file locks are released
        time.sleep(1)

    # 5. GENERATE FINAL REPORT
    print("\n\n==================================================")
    print("📝 GENERATING RESEARCH REPORT (Azure OpenAI)")
    print("==================================================")
    
    if campaign_history:
        # Send the full history to GPT-4o
        final_report = scientist.generate_campaign_summary(campaign_history)
        print(final_report)
        
        # Save report to the Lab Notebook folder
        report_path = os.path.join(lab_notebook.base_path, "Final_Scientific_Report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(final_report)
        print(f"\n✅ Report saved to: {report_path}")
        print(f"📂 Full Data & Structures saved in: {lab_notebook.base_path}")
    else:
        print("❌ No discoveries made.")

    print("\n--- 🏁 CAMPAIGN FINISHED ---")

if __name__ == "__main__":
    # You can adjust the number of generations here
    run_hybrid_discovery(generations=5)