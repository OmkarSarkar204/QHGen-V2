import os
import sys
from openai import AzureOpenAI

class AIchemist:
    def __init__(self):
        # 🔑 PASTE YOUR KEY HERE
        self.api_key = "FeqYXleMfVBrM7uA5CB5yJMXjFTCpKhTgYIT910QJgpimP2ymsMuJQQJ99BLACHYHv6XJ3w3AAAAACOGhYBm"
        self.endpoint = "https://omkarsarkar204-8826-resource.openai.azure.com/"
        
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint, 
            api_key=self.api_key, 
            api_version="2024-02-15-preview"
        )
        self.deployment_name = "gpt-4o" 

    def generate_campaign_summary(self, history_list):
        """
        Analyzes the entire campaign history to write a final summary.
        """
        print("... Sending campaign data to GPT-4o (Timeout: 45s) ...")
        
        # Format the data for the LLM
        data_str = ""
        for item in history_list:
            comp_str = ", ".join([f"{k}: {v:.2f}" for k,v in item['composition'].items()])
            # Handle cases where 'uncertainty' might be missing from older logs
            unc = item.get('uncertainty', 0.0)
            data_str += f"- Gen {item['generation']}: {{{comp_str}}} | Quantum Energy: {item['energy']:.4f} eV | Uncertainty: {unc:.4f}\n"

        prompt = f"""
        You are the Chief Scientist of the QHGen Project.
        TIMELINE OF DISCOVERIES:
        {data_str}
        
        TASK:
        1. **Trend Analysis:** Did the AI improve?
        2. **Best Candidate:** Identify the best alloy (Energy closest to 0.0 eV).
        3. **Mechanism:** Explain scientifically why this ratio works.
        4. **Lab Recommendation:** Write a synthesis recipe.
        
        Format as a professional Markdown report.
        """

        try:
            # ⚡ FIX: Added timeout=45 to prevent hanging forever
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert material science consultant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7,
                timeout=45 
            )
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"⚠️ OpenAI Error: {e}")
            return f"❌ Report Generation Failed due to API Error: {e}"

# --- 🧪 TEST LINE (RUN THIS FILE DIRECTLY) ---
if __name__ == "__main__":
    print("🔬 TESTING AZURE OPENAI CONNECTION...")
    
    # Fake data to test the LLM
    dummy_history = [
        {'generation': 1, 'composition': {'Ni': 0.5, 'Fe': 0.5}, 'energy': 0.15, 'uncertainty': 0.05},
        {'generation': 2, 'composition': {'Ni': 0.9, 'Fe': 0.1}, 'energy': 0.02, 'uncertainty': 0.01}
    ]
    
    scientist = AIchemist()
    
    if scientist.api_key == "PASTE_YOUR_KEY_HERE":
        print("❌ ERROR: You forgot to paste your API Key in line 8!")
    else:
        report = scientist.generate_campaign_summary(dummy_history)
        print("\n--- RECEIVED REPORT ---")
        print(report)
        print("-----------------------")