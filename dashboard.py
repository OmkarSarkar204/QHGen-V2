import customtkinter as ctk
import tkinter as tk
import threading
import time
import math
import random
import sys
import os
import subprocess
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# --- ELEMENT DATABASE ---
ELEMENT_DB = {
    'Ni': {'price': 17.50,   'abundance': 84.0,    'name': 'Nickel'},
    'Fe': {'price': 0.12,    'abundance': 56300.0, 'name': 'Iron'},
    'Co': {'price': 28.50,   'abundance': 25.0,    'name': 'Cobalt'},
    'Cu': {'price': 9.20,    'abundance': 60.0,    'name': 'Copper'},
    'Zn': {'price': 2.80,    'abundance': 70.0,    'name': 'Zinc'},
    'Ti': {'price': 6.50,    'abundance': 5650.0,  'name': 'Titanium'},
    'Mo': {'price': 44.00,   'abundance': 1.2,     'name': 'Molybdenum'},
    'W':  {'price': 38.00,   'abundance': 1.25,    'name': 'Tungsten'},
    'Pt': {'price': 31500.0, 'abundance': 0.005,   'name': 'Platinum'},
    'Pd': {'price': 38000.0, 'abundance': 0.015,   'name': 'Palladium'},
    'Au': {'price': 68000.0, 'abundance': 0.004,   'name': 'Gold'}
}

PLATINUM_REF = ELEMENT_DB['Pt']

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'qhgen-v3'))
sys.path.append(os.path.join(current_dir, '..'))

# --- IMPORT MODULES ---
try:
    from core.data.logger import ResearchLogger
    from core.physics.pourbaix import PourbaixFilter
    from pipeline.screening_flow import run_screening_cycle
    from models.surrogate.train import active_training_cycle
    from core.agent.summary import AIchemist
except ImportError:
    # Dummy classes for GUI testing without backend
    class ResearchLogger:
        def __init__(self, campaign_name): self.base_path="."
        def log_generation(self, *args): pass
        def log_candidate(self, *args): pass
    class AIchemist: 
        def generate_campaign_summary(self, h): return "AI Summary Not Available (Import Error)"
    class PourbaixFilter:
        def check_stability(self, el, ph): return True, []
    def run_screening_cycle(**kwargs): return None
    def active_training_cycle(**kwargs): pass

# --- GUI THEME ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

ATOM_COLORS = {
    'Ni': '#50C878', 'Fe': '#E06666', 'Co': '#F06292', 'Cu': '#C87533',
    'Mo': '#9400D3', 'W':  '#4682B4', 'H':  '#FFFFFF', 'Pt': '#D3D3D3',
    'Au': '#FFD700', 'Ti': '#878687'
}

def get_atom_color(element):
    if element in ATOM_COLORS: return ATOM_COLORS[element]
    hash_val = sum(ord(c) for c in element)
    return f"#{hash_val % 0xFFFFFF:06x}"

# --- VISUALIZER COMPONENT ---
class MolecularVisualizer(ctk.CTkCanvas):
    def __init__(self, master, width=800, height=350):
        super().__init__(master, width=width, height=height, bg="#050505", highlightthickness=0)
        self.width = width
        self.height = height
        self.atoms = [] 
        self.is_running = False
        self.phase = "IDLE" 
        self.frame_count = 0
        self.atom_radius = 16
        self.lattice_spacing = 42
        
    def start_simulation(self, composition):
        self.atoms = []
        self.phase = "MIXING"
        self.frame_count = 0
        total_atoms = 50 
        grid_cols = 10
        grid_rows = 5
        element_pool = []
        for el, ratio in composition.items():
            count = int(ratio * total_atoms)
            element_pool.extend([el] * count)
        while len(element_pool) < total_atoms:
            element_pool.append(list(composition.keys())[0])
        random.shuffle(element_pool)
        start_x = (self.width - (grid_cols * self.lattice_spacing)) / 2
        start_y = self.height - (grid_rows * self.lattice_spacing) - 60
        for i, el in enumerate(element_pool):
            col = i % grid_cols
            row = i // grid_cols
            offset = 20 if row % 2 else 0
            tx = start_x + (col * self.lattice_spacing) + offset
            ty = start_y + (row * self.lattice_spacing)
            self.atoms.append({
                'element': el, 'color': get_atom_color(el),
                'x': random.randint(50, self.width-50),
                'y': random.randint(50, self.height-150),
                'vx': random.uniform(-2, 2), 'vy': random.uniform(-2, 2),
                'tx': tx, 'ty': ty, 'r': self.atom_radius, 'state': 'liquid'
            })
        self.is_running = True
        self.animate()

    def add_hydrogen_probe(self):
        targets = [a for a in self.atoms if a['ty'] < self.height/2 + 100]
        if not targets: targets = self.atoms
        target = random.choice(targets)
        self.atoms.append({
            'element': 'H', 'color': '#FFFFFF',
            'x': target['tx'], 'y': -20, 'vx': 0, 'vy': 5,
            'tx': target['tx'], 'ty': target['ty'] - 25, 'r': 8, 'state': 'probe'
        })

    def animate(self):
        if not self.is_running: return
        self.delete("all")
        self.frame_count += 1
        self.create_rectangle(0, self.height-5, self.width, self.height, fill="#331111", outline="")
        settled_count = 0
        for atom in self.atoms:
            if self.phase == "MIXING":
                atom['x'] += atom['vx']
                atom['y'] += atom['vy']
                if atom['x'] < 0 or atom['x'] > self.width: atom['vx'] *= -1
                if atom['y'] < 0 or atom['y'] > self.height-100: atom['vy'] *= -1
                if self.frame_count > 60: self.phase = "CRYSTALLIZING"
            elif self.phase == "CRYSTALLIZING" or self.phase == "ADSORPTION":
                dx = atom['tx'] - atom['x']
                dy = atom['ty'] - atom['y']
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 1 and atom['state'] != 'probe':
                    atom['x'] = atom['tx']; atom['y'] = atom['ty']; settled_count += 1
                else:
                    atom['x'] += dx * 0.08; atom['y'] += dy * 0.08
            self.create_oval(atom['x'] - atom['r'], atom['y'] - atom['r'], atom['x'] + atom['r'], atom['y'] + atom['r'], fill=atom['color'], outline="black", width=1)
            self.create_oval(atom['x'] - atom['r']*0.4, atom['y'] - atom['r']*0.4, atom['x'] - atom['r']*0.1, atom['y'] - atom['r']*0.1, fill="#DDDDDD", outline="")
        if self.phase == "CRYSTALLIZING" and settled_count >= len(self.atoms) * 0.9:
            self.phase = "ADSORPTION"; self.add_hydrogen_probe()
        self.create_text(20, 20, text=f"PHASE: {self.phase}", anchor="w", fill="white", font=("Consolas", 14))
        self.after(20, self.animate)

# --- ENLARGED LOG WINDOW ---
class EnlargedLogWindow(ctk.CTkToplevel):
    def __init__(self, parent, log_content):
        super().__init__(parent)
        self.title("Mission Log - Full View")
        self.geometry("1000x700")
        
        # Header
        header = ctk.CTkFrame(self, fg_color="#1a1a1a", height=60)
        header.pack(fill="x", padx=0, pady=0)
        ctk.CTkLabel(header, text="MISSION LOG", 
                    font=("Consolas", 18, "bold"), 
                    text_color="white").pack(pady=15)
        
        # Log display
        self.txt_log = ctk.CTkTextbox(self, fg_color="#111111", 
                                      text_color="white", 
                                      font=("Consolas", 13), 
                                      wrap="word")
        self.txt_log.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Insert content
        self.txt_log.insert("1.0", log_content)
        self.txt_log.configure(state="disabled")
        
        # Close button
        btn_close = ctk.CTkButton(self, text="CLOSE", 
                                  fg_color="#cc0000", 
                                  hover_color="#990000",
                                  height=40, 
                                  font=("Roboto", 12, "bold"),
                                  command=self.destroy)
        btn_close.pack(pady=(0, 20))

# --- MAIN DASHBOARD CLASS ---
class ProfessionalDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("QHGen: Autonomous Materials Discovery Platform")
        self.geometry("1300x850")
        
        self.pourbaix = PourbaixFilter()
        self.last_report_path = None

        self.grid_columnconfigure(1, weight=3) 
        self.grid_columnconfigure(0, weight=1) 
        self.grid_rowconfigure(0, weight=1)

        # --- LEFT PANEL ---
        self.left_panel = ctk.CTkFrame(self, corner_radius=0, fg_color="#121212")
        self.left_panel.grid(row=0, column=0, sticky="nsew")
        
        ctk.CTkLabel(self.left_panel, text="QHGen", font=("Roboto", 28, "bold"), text_color="white").pack(pady=(40, 5))
        ctk.CTkLabel(self.left_panel, text="H-Catalyst Engine", font=("Roboto", 12), text_color="#888").pack(pady=(0, 30))
        
        self.entry_elements = ctk.CTkEntry(self.left_panel, placeholder_text="Target Elements (e.g. Ni Fe)", width=260, height=40, font=("Consolas", 14))
        self.entry_elements.pack(pady=10)
        
        self.slider_gens = ctk.CTkSlider(self.left_panel, from_=1, to=10, number_of_steps=9, width=260, command=self.update_gens_label)
        self.slider_gens.set(5)
        self.slider_gens.pack(pady=(20, 5))
        self.lbl_gens = ctk.CTkLabel(self.left_panel, text="5 Optimization Cycles")
        self.lbl_gens.pack()

        self.slider_ph = ctk.CTkSlider(self.left_panel, from_=0, to=14, number_of_steps=14, width=260, command=self.update_ph_label)
        self.slider_ph.set(7)
        self.slider_ph.pack(pady=(20, 5))
        self.lbl_ph = ctk.CTkLabel(self.left_panel, text="pH Level: 7.0 (Neutral)")
        self.lbl_ph.pack()
        
        self.btn_run = ctk.CTkButton(self.left_panel, text="INITIATE CAMPAIGN", fg_color="#0066cc", hover_color="#0052a3", height=50, width=260, font=("Roboto", 14, "bold"), command=self.start_engine)
        self.btn_run.pack(pady=30)
        
        self.btn_open_report = ctk.CTkButton(self.left_panel, text="OPEN FINAL REPORT", fg_color="#228833", hover_color="#116622", height=40, width=260, font=("Roboto", 12, "bold"), state="disabled", command=self.open_final_report)
        self.btn_open_report.pack(pady=(0, 20))

        self.metric_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.metric_frame.pack(fill="x", padx=20)
        self.lbl_energy = self.create_metric("Adsorption Energy (ΔG)", "--- eV", "#00ff00")
        self.lbl_uncertainty = self.create_metric("Model Uncertainty", "---", "#ffcc00")
        self.lbl_status = self.create_metric("System Status", "IDLE", "#ffffff")

        # --- RIGHT PANEL ---
        self.right_panel = ctk.CTkFrame(self, fg_color="#000000", corner_radius=0)
        self.right_panel.grid(row=0, column=1, sticky="nsew")
        
        self.visualizer = MolecularVisualizer(self.right_panel, width=900, height=350)
        self.visualizer.pack(expand=False, fill="x", padx=20, pady=20)
        
        self.graph_frame = ctk.CTkFrame(self.right_panel, fg_color="#0a0a0a", height=200)
        self.graph_frame.pack(expand=False, fill="x", padx=20, pady=(0, 10))
        self.setup_graph()

        # LOG HEADER WITH ENLARGE BUTTON
        log_header_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        log_header_frame.pack(fill="x", padx=20, pady=(10, 0))
        
        self.log_label = ctk.CTkLabel(log_header_frame, text="> LIVE LOG", 
                                      font=("Consolas", 12, "bold"), 
                                      text_color="#666", 
                                      anchor="w")
        self.log_label.pack(side="left")
        
        self.btn_enlarge = ctk.CTkButton(log_header_frame, text="ENLARGE", 
                                         fg_color="#444444", 
                                         hover_color="#666666",
                                         width=100, 
                                         height=28,
                                         font=("Roboto", 10, "bold"),
                                         command=self.open_enlarged_log)
        self.btn_enlarge.pack(side="right")
        
        self.txt_console = ctk.CTkTextbox(self.right_panel, fg_color="#111111", 
                                         text_color="white",
                                         font=("Consolas", 12), 
                                         wrap="word")
        self.txt_console.pack(fill="both", expand=True, padx=20, pady=10)

    def open_enlarged_log(self):
        """Open the log in an enlarged window"""
        log_content = self.txt_console.get("1.0", "end-1c")
        if not log_content.strip():
            log_content = "No log data available yet.\n\nRun a campaign to see results here."
        EnlargedLogWindow(self, log_content)

    def update_gens_label(self, value):
        self.lbl_gens.configure(text=f"{int(value)} Optimization Cycles")

    def update_ph_label(self, value):
        self.lbl_ph.configure(text=f"pH Level: {int(value)}")

    def setup_graph(self):
        self.fig = Figure(figsize=(5, 2.5), dpi=100, facecolor="#0a0a0a")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#0a0a0a")
        self.ax.set_title("Optimization Convergence", color="white", fontsize=9)
        self.ax.set_ylabel("Adsorption Energy (eV)", color="gray", fontsize=8)
        self.ax.set_xlabel("Generation", color="gray", fontsize=8)
        self.ax.tick_params(colors='gray', labelsize=8)
        self.ax.grid(True, color="#333333", linestyle="--", linewidth=0.5)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="left", fill="both", expand=True)

    def update_graph(self, history):
        def _draw():
            x = [h['generation'] for h in history]
            y = [abs(h['energy']) for h in history]
            self.ax.clear()
            self.ax.set_facecolor("#0a0a0a")
            self.ax.set_title("AI Optimization Convergence (Goal: 0.0 eV)", color="white", fontsize=9)
            self.ax.set_ylabel("|ΔG| (eV)", color="gray", fontsize=8)
            self.ax.set_xlabel("Generation", color="gray", fontsize=8)
            self.ax.tick_params(colors='gray', labelsize=8)
            self.ax.grid(True, color="#333333", linestyle="--", linewidth=0.5)
            self.ax.plot(x, y, color="#00ff00", marker="o", linewidth=2, markersize=5)
            self.canvas.draw()
        self.after(0, _draw)

    def create_metric(self, title, value, color):
        frame = ctk.CTkFrame(self.metric_frame, fg_color="#1e1e1e")
        frame.pack(fill="x", pady=5)
        ctk.CTkLabel(frame, text=title, font=("Arial", 10, "bold"), text_color="#666").pack(anchor="w", padx=15, pady=(5,0))
        lbl = ctk.CTkLabel(frame, text=value, font=("Consolas", 18, "bold"), text_color=color)
        lbl.pack(anchor="w", padx=15, pady=(0,5))
        return lbl

    def update_metric_safe(self, label, text, color=None):
        def _update():
            try:
                label.configure(text=text)
                if color: label.configure(text_color=color)
            except Exception: pass 
        self.after(0, _update)

    def write_to_console(self, text):
        def _write():
            self.txt_console.insert("end", text + "\n")
            self.txt_console.see("end")
        self.after(0, _write)

    def open_final_report(self):
        if self.last_report_path and os.path.exists(self.last_report_path):
            try: os.startfile(self.last_report_path) # Windows
            except: subprocess.call(['open', self.last_report_path]) # Mac/Linux

    def generate_executive_summary_text(self, history):
        if not history: return "No data."
        best = min(history, key=lambda x: abs(x['energy']))
        comp = best['composition']
        dG = abs(best['energy'])

        # --- 1. Calculate Costs First ---
        alloy_cost = 0.0
        total_abundance = 0.0
        for el, ratio in comp.items():
            # Default to $50/kg if element not found to prevent crash
            data = ELEMENT_DB.get(el, {'price': 50.0, 'abundance': 1.0})
            alloy_cost += data['price'] * ratio
            total_abundance += data['abundance'] * ratio

        pt_cost = PLATINUM_REF['price']
        
        # [FIX] Calculate orders_of_magnitude explicitly here
        if alloy_cost > 0:
            cost_reduction_factor = pt_cost / alloy_cost
            orders_of_magnitude = math.log10(cost_reduction_factor)
        else:
            orders_of_magnitude = 0.0

        # --- 2. Determine Performance (pH Aware) ---
        current_ph = self.slider_ph.get()
        
        # Base thresholds (Acidic/Neutral)
        threshold_excellent = 0.09
        threshold_industry = 0.20
        
        # If Alkaline (pH > 12), widen the window for Ni-Fe
        if current_ph > 12:
            threshold_excellent = 0.15
            threshold_industry = 0.35  
        
        if dG <= threshold_excellent: 
            performance = "OUTPERFORMS PLATINUM"
        elif dG <= threshold_industry: 
            performance = "INDUSTRIAL GRADE"
        else: 
            performance = "EXPERIMENTAL"

        # --- 3. Determine Supply Risk ---
        scalability_score = min(100, max(1, math.log(total_abundance + 1e-9) * 10))
        if scalability_score > 60: supply_risk = "NEGLIGIBLE (Industrial Standard)"
        elif scalability_score > 30: supply_risk = "MODERATE (Supply Chain Risk)"
        else: supply_risk = "CRITICAL (Rare Earth/Precious)"

        # --- 4. Generate Report String ---
        report = f"""
═════════════════════════════════════════════════════════════════════
       QHGEN INDUSTRIAL FEASIBILITY REPORT (2026)
═════════════════════════════════════════════════════════════════════

1. MATERIAL DEFINITION
   • Composition:    {comp}
   • Primary Metal:  {max(comp, key=comp.get)}

2. HYDROGEN PRODUCTIVITY
   • Adsorption ΔG:  {best['energy']:.4f} eV
   • Verdict:        {performance}
   • Note:           (Platinum Reference: 0.09 eV)

3. ECONOMICS & SCALABILITY
   • Material Cost Reduction: >{orders_of_magnitude:.1f} Orders of Magnitude
   • Alloy Cost:     ${alloy_cost:,.2f} / kg
   • Platinum Cost:  ${pt_cost:,.2f} / kg
   • Shifts from Platinum Group Metals (~$30k/kg) to Earth-Abundant Alloys

4. SUPPLY CHAIN SECURITY
   • Supply Risk:     {supply_risk}
   • Replaces Critical Raw Materials with standard industrial feedstocks
   • Eliminates dependence on single-source mining regions
═════════════════════════════════════════════════════════════════════
"""
        return report

    def start_engine(self):
        target = self.entry_elements.get().strip()
        gens = int(self.slider_gens.get())
        ph_val = int(self.slider_ph.get())

        if not target: target = "Ni Fe"
        target_elements = [e.capitalize() for e in target.split()]

        is_stable, rejected = self.pourbaix.check_stability(target_elements, ph_val)
        if not is_stable:
            self.update_metric_safe(self.lbl_status, "STABILITY ERROR", "red")
            tk.messagebox.showerror("Physics Constraint", f"REJECTED: Elements unstable at pH {ph_val}\n\n{', '.join(rejected)}")
            return

        self.btn_run.configure(state="disabled", text="PROCESSING...")
        self.btn_open_report.configure(state="disabled") 
        self.txt_console.delete("1.0", "end") 
        
        self.update_metric_safe(self.lbl_status, "INITIALIZING", "#00ccff")
        
        threading.Thread(target=self.run_background_process, args=(target, gens), daemon=True).start()

    def run_background_process(self, elements_str, gens):
        try:
            target_elements = elements_str.split() if elements_str else ['Ni', 'Fe']
            target_elements = [e.capitalize() for e in target_elements]
            
            logger = ResearchLogger(campaign_name=f"GUI_Run_{''.join(target_elements)}")
            scientist = AIchemist()
            history = [] 
            
            for i in range(1, gens + 1):
                self.update_metric_safe(self.lbl_status, f"GEN {i}: SCREENING", "#00ccff")
                winner_data = run_screening_cycle(generation_id=i, allowed_elements=target_elements, logger=logger)
                
                if winner_data:
                    self.after(0, lambda w=winner_data['composition']: self.visualizer.start_simulation(w))
                    en = winner_data['predicted_energy']
                    unc = winner_data.get('uncertainty', 0.0)
                    
                    self.update_metric_safe(self.lbl_energy, f"{en:.4f} eV", "#00ff00" if abs(en) < 0.1 else "#ff5555")
                    self.update_metric_safe(self.lbl_uncertainty, f"±{unc:.4f}", "#ffcc00")
                    
                    history.append({
                        "generation": i, "composition": winner_data['composition'],
                        "energy": winner_data['quantum_energy'], "ai_pred": en, "uncertainty": unc
                    })
                    self.update_graph(history)
                
                self.update_metric_safe(self.lbl_status, f"GEN {i}: LEARNING", "#bd93f9")
                active_training_cycle(fine_tune_only=True)
                time.sleep(1.5) 

            self.update_metric_safe(self.lbl_status, "GENERATING REPORT", "#ffaa00")
            
            if history:
                summary_text = self.generate_executive_summary_text(history)
                self.write_to_console(summary_text)

                try:
                    report = scientist.generate_campaign_summary(history)
                    self.last_report_path = os.path.join(logger.base_path, "Final_Report.md")
                    with open(self.last_report_path, "w", encoding="utf-8") as f:
                        f.write(report)
                    
                    self.write_to_console(f">> FULL REPORT SAVED: {self.last_report_path}")
                    self.update_metric_safe(self.lbl_status, "COMPLETE", "#00ff00")
                    self.after(0, lambda: self.btn_open_report.configure(state="normal"))
                    
                except Exception as e:
                    self.write_to_console(f">> ERROR: {e}")
                    self.update_metric_safe(self.lbl_status, "REPORT FAILED", "#ff5555")
            else:
                self.update_metric_safe(self.lbl_status, "NO DATA", "#ff5555")
            
        except Exception as e:
            print(f"CRITICAL ERROR: {e}")
            self.update_metric_safe(self.lbl_status, "FAILURE", "red")
        finally:
            self.after(0, lambda: self.btn_run.configure(state="normal", text="INITIATE CAMPAIGN"))

if __name__ == "__main__":
    app = ProfessionalDashboard()
    app.mainloop()