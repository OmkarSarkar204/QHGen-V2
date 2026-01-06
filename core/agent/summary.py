from openai import AzureOpenAI

class AIchemist:
    def __init__(self):
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
        Analyzes the entire campaign history to write a final technical report.
        """
        print("... Sending campaign data to GPT-4o (Timeout: 45s) ...")
        
        data_str = ""
        for item in history_list:
            comp_str = ", ".join([f"{k}: {v:.2f}" for k,v in item['composition'].items()])
            unc = item.get('uncertainty', 0.0)
            data_str += f"- Gen {item['generation']}: {{{comp_str}}} | Delta G: {item['energy']:.4f} eV | Uncertainty: {unc:.4f}\n"

        prompt = f"""
        ACT AS: Automated Research Logger (QHGen System).
        CONTEXT: High-Throughput Screening for Hydrogen Evolution Reaction (HER) Catalysts.
        
        RAW SIMULATION DATA:
        {data_str}
        
        TASK: Generate a "Telegraphic Style" Scientific Log.
        CONSTRAINTS:
        1. NO conversational filler (e.g., "Here is the report", "In conclusion").
        2. Tone: Clinical, sparse, data-driven.
        3. Format: Strict Markdown.
        
        REQUIRED SECTIONS:
        
        ### 1. CONVERGENCE ANALYSIS
        [Briefly state if the genetic algorithm converged on a stable minimum. 1 sentence.]
        
        ### 2. OPTIMAL CANDIDATE SPECIFICATION
        * **Composition:** [Insert Best Formula]
        * **Binding Energy (ΔG):** [Insert Value] eV
        * **Performance vs Platinum:** [Compare closest candidate to Pt (-0.09 eV)]
        
        ### 3. THEORETICAL MECHANISM (CLARIFICATION)
        [Explain strictly *why* this works using electronic structure theory. Mention d-band center shifting, lattice strain, or ligand effects. Do not fluff.]
        
        ### 4. SYNTHESIS PARAMETERS (PROPOSED)
        [Do not write instructions. List technical parameters only.]
        * **Method:** [e.g., Sol-Gel / Ball Milling / Electrodeposition]
        * **Precursors:** [List likely metal salts]
        * **Annealing Temp:** [Estimated C]
        * **Atmosphere:** [e.g., Ar/H2 flow]
        """

        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a scientific data logger. Output raw data and facts only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.5,
                timeout=45 
            )
            return response.choices[0].message.content
            
        except Exception as e:
            print(f" OpenAI Error: {e}")
            return f" Report Generation Failed due to API Error: {e}"

if __name__ == "__main__":
    print(" TESTING AZURE OPENAI CONNECTION...")
    
    dummy_history = [
        {'generation': 1, 'composition': {'Ni': 0.5, 'Fe': 0.5}, 'energy': 0.15, 'uncertainty': 0.05},
        {'generation': 2, 'composition': {'Ni': 0.9, 'Fe': 0.1}, 'energy': 0.02, 'uncertainty': 0.01}
    ]
    
    scientist = AIchemist()
    
    if "PASTE" in scientist.api_key:
        print(" ERROR: You forgot to paste your API Key in the __init__ function!")
    else:
        report = scientist.generate_campaign_summary(dummy_history)
        print("\n--- RECEIVED REPORT ---")
        print(report)
        print("-----------------------")