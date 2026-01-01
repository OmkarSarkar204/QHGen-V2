import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import threading
import time
import math
import random
import sys
import os

# --- MATPLOTLIB IMPORTS ---
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'qhgen-v3'))
sys.path.append(os.path.join(current_dir, '..'))

try:
    from core.data.logger import ResearchLogger
    from pipeline.screening_flow import run_screening_cycle
    from models.surrogate.train import active_training_cycle
    from core.agent.summary import AIchemist
except ImportError:
    class ResearchLogger: 
        def __init__(self, campaign_name): self.base_path="."
    class AIchemist: pass
    def run_screening_cycle(**kwargs): return None
    def active_training_cycle(**kwargs): pass

# --- CONFIGURATION ---
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

ATOM_COLORS = {
    'Ni': '#50C878', 'Fe': '#E06666', 'Co': '#F06292', 'Cu': '#C87533',
    'Mo': '#9400D3', 'W':  '#4682B4', 'H':  '#FFFFFF', 'Pt': '#D3D3D3',
    'Au': '#FFD700', 'Ti': '#878687'
}

def get_atom_color(element):
    if element in ATOM_COLORS: return ATOM_COLORS[element]
    hash_val = sum(ord(c) for c in element)
    return f"#{hash_val % 0xFFFFFF:06x}"

class MolecularVisualizer(ctk.CTkCanvas):
    def __init__(self, master, width=800, height=400):
        super().__init__(master, width=width, height=height, bg="#f5f5f5", highlightthickness=1, highlightbackground="#cccccc")
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
        self.create_rectangle(0, self.height-5, self.width, self.height, fill="#e0e0e0", outline="")
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
            self.create_oval(atom['x'] - atom['r'], atom['y'] - atom['r'], atom['x'] + atom['r'], atom['y'] + atom['r'], fill=atom['color'], outline="#333333", width=1)
            self.create_oval(atom['x'] - atom['r']*0.4, atom['y'] - atom['r']*0.4, atom['x'] - atom['r']*0.1, atom['y'] - atom['r']*0.1, fill="#FFFFFF", outline="")
        if self.phase == "CRYSTALLIZING" and settled_count >= len(self.atoms) * 0.9:
            self.phase = "ADSORPTION"; self.add_hydrogen_probe()
        self.create_text(20, 20, text=f"Phase: {self.phase}", anchor="w", fill="#333333", font=("Arial", 11))
        self.after(20, self.animate)

class LaboratoryInterface(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("QHGen Materials Discovery Platform v3.0")
        self.geometry("1400x900")
        
        # Main container
        main_container = ctk.CTkFrame(self, fg_color="#ffffff")
        main_container.pack(fill="both", expand=True, padx=0, pady=0)
        
        # Top: Header Bar
        header = ctk.CTkFrame(main_container, height=70, fg_color="#2c3e50", corner_radius=0)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)
        
        ctk.CTkLabel(header, text="QHGen v3.0", font=("Arial", 22, "bold"), text_color="#ffffff").pack(side="left", padx=30, pady=20)
        ctk.CTkLabel(header, text="Quantum-Hybrid Materials Discovery Platform", font=("Arial", 11), text_color="#bdc3c7").pack(side="left", padx=(0, 20), pady=20)
        
        # Middle: Content area
        content = ctk.CTkFrame(main_container, fg_color="#ffffff")
        content.pack(fill="both", expand=True, padx=0, pady=0)
        
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=2)
        content.grid_rowconfigure(0, weight=1)
        
        # LEFT PANEL: Controls
        left_panel = ctk.CTkFrame(content, fg_color="#ecf0f1", corner_radius=0)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        
        # Control Section
        control_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
        control_frame.pack(fill="x", padx=25, pady=(30, 20))
        
        ctk.CTkLabel(control_frame, text="Experimental Parameters", font=("Arial", 14, "bold"), text_color="#2c3e50").pack(anchor="w", pady=(0, 15))
        
        ctk.CTkLabel(control_frame, text="Target Elements", font=("Arial", 10), text_color="#34495e").pack(anchor="w", pady=(0, 5))
        self.entry_elements = ctk.CTkEntry(control_frame, placeholder_text="Ni Fe Co", width=280, height=38, font=("Consolas", 12), fg_color="#ffffff", border_color="#bdc3c7")
        self.entry_elements.pack(pady=(0, 20))
        
        ctk.CTkLabel(control_frame, text="Optimization Cycles", font=("Arial", 10), text_color="#34495e").pack(anchor="w", pady=(0, 5))
        self.slider_gens = ctk.CTkSlider(control_frame, from_=1, to=10, number_of_steps=9, width=280, button_color="#3498db", button_hover_color="#2980b9", progress_color="#3498db")
        self.slider_gens.set(5)
        self.slider_gens.pack(pady=(0, 5))
        self.lbl_gens = ctk.CTkLabel(control_frame, text="5 Cycles", font=("Arial", 10), text_color="#7f8c8d")
        self.lbl_gens.pack(anchor="w")
        
        self.slider_gens.configure(command=lambda v: self.lbl_gens.configure(text=f"{int(v)} Cycles"))
        
        self.btn_run = ctk.CTkButton(control_frame, text="Start Campaign", fg_color="#27ae60", hover_color="#229954", height=45, width=280, font=("Arial", 13, "bold"), command=self.start_engine, corner_radius=5)
        self.btn_run.pack(pady=(25, 0))
        
        # Metrics Section
        metrics_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
        metrics_frame.pack(fill="both", expand=True, padx=25, pady=(20, 30))
        
        ctk.CTkLabel(metrics_frame, text="Real-Time Metrics", font=("Arial", 14, "bold"), text_color="#2c3e50").pack(anchor="w", pady=(0, 15))
        
        self.metric_energy = self.create_metric_panel(metrics_frame, "Adsorption Energy (ΔG)", "--- eV")
        self.metric_uncertainty = self.create_metric_panel(metrics_frame, "Model Uncertainty (σ)", "---")
        self.metric_composition = self.create_metric_panel(metrics_frame, "Current Composition", "---")
        
        # Status Bar
        status_container = ctk.CTkFrame(left_panel, fg_color="#34495e", height=50, corner_radius=0)
        status_container.pack(fill="x", side="bottom", padx=0, pady=0)
        status_container.pack_propagate(False)
        
        self.lbl_status = ctk.CTkLabel(status_container, text="System Status: IDLE", font=("Arial", 11, "bold"), text_color="#ecf0f1")
        self.lbl_status.pack(pady=15, padx=20)
        
        # RIGHT PANEL: Visualizations
        right_panel = ctk.CTkFrame(content, fg_color="#ffffff", corner_radius=0)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        
        # Molecular Viewer (Top)
        viewer_section = ctk.CTkFrame(right_panel, fg_color="transparent")
        viewer_section.pack(fill="both", expand=False, padx=25, pady=(25, 15))
        
        ctk.CTkLabel(viewer_section, text="Structural Visualization", font=("Arial", 12, "bold"), text_color="#2c3e50").pack(anchor="w", pady=(0, 10))
        self.visualizer = MolecularVisualizer(viewer_section, width=850, height=380)
        self.visualizer.pack()
        
        # Analysis Graph (Bottom)
        graph_section = ctk.CTkFrame(right_panel, fg_color="transparent")
        graph_section.pack(fill="both", expand=True, padx=25, pady=(15, 25))
        
        ctk.CTkLabel(graph_section, text="Optimization Convergence Analysis", font=("Arial", 12, "bold"), text_color="#2c3e50").pack(anchor="w", pady=(0, 10))
        
        graph_container = ctk.CTkFrame(graph_section, fg_color="#f8f9fa", corner_radius=5)
        graph_container.pack(fill="both", expand=True)
        
        self.setup_graph(graph_container)

    def create_metric_panel(self, parent, title, initial_value):
        frame = ctk.CTkFrame(parent, fg_color="#ffffff", corner_radius=5, border_width=1, border_color="#d5d8dc")
        frame.pack(fill="x", pady=(0, 12))
        
        title_label = ctk.CTkLabel(frame, text=title, font=("Arial", 9), text_color="#7f8c8d", anchor="w")
        title_label.pack(anchor="w", padx=15, pady=(12, 2))
        
        value_label = ctk.CTkLabel(frame, text=initial_value, font=("Consolas", 16, "bold"), text_color="#2c3e50", anchor="w")
        value_label.pack(anchor="w", padx=15, pady=(0, 12))
        
        return value_label

    def setup_graph(self, parent):
        self.fig = Figure(figsize=(7, 3.5), dpi=100, facecolor="#f8f9fa")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#ffffff")
        
        self.ax.set_title("Energy Minimization Progress", color="#2c3e50", fontsize=11, fontweight="bold", pad=12)
        self.ax.set_ylabel("Energy (eV)", color="#34495e", fontsize=9)
        self.ax.set_xlabel("Generation", color="#34495e", fontsize=9)
        self.ax.tick_params(colors='#7f8c8d', labelsize=8)
        self.ax.grid(True, color="#e0e0e0", linestyle="-", linewidth=0.5, alpha=0.7)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color('#bdc3c7')
        self.ax.spines['bottom'].set_color('#bdc3c7')
        
        self.fig.tight_layout(pad=2.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=15, pady=15)

    def update_graph(self, history):
        if not history:
            return
            
        x = [h['generation'] for h in history]
        y = [h['quantum_energy'] for h in history]
        
        self.ax.clear()
        
        self.ax.set_facecolor("#ffffff")
        self.ax.set_title("Energy Minimization Progress", color="#2c3e50", fontsize=11, fontweight="bold", pad=12)
        self.ax.set_ylabel("Adsorption Energy (eV)", color="#34495e", fontsize=9)
        self.ax.set_xlabel("Generation", color="#34495e", fontsize=9)
        self.ax.tick_params(colors='#7f8c8d', labelsize=8)
        self.ax.grid(True, color="#e0e0e0", linestyle="-", linewidth=0.5, alpha=0.7)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color('#bdc3c7')
        self.ax.spines['bottom'].set_color('#bdc3c7')
        
        self.ax.plot(x, y, color="#3498db", marker="o", linewidth=2, markersize=6, markerfacecolor="#2980b9", markeredgecolor="#2c3e50", markeredgewidth=0.5)
        
        if len(y) > 1:
            self.ax.axhline(y=0, color="#95a5a6", linestyle="--", linewidth=1, alpha=0.6)
        
        self.fig.tight_layout(pad=2.0)
        self.canvas.draw()

    def update_metric_safe(self, label, text):
        label.configure(text=text)

    def start_engine(self):
        target = self.entry_elements.get().strip()
        gens = int(self.slider_gens.get())
        
        self.btn_run.configure(state="disabled", text="Running...", fg_color="#95a5a6")
        self.lbl_status.configure(text="System Status: INITIALIZING")
        
        threading.Thread(target=self.run_background_process, args=(target, gens), daemon=True).start()

    def run_background_process(self, elements_str, gens):
        try:
            target_elements = elements_str.split() if elements_str else ['Ni', 'Fe']
            target_elements = [e.capitalize() for e in target_elements]
            
            logger = ResearchLogger(campaign_name=f"Campaign_{''.join(target_elements)}")
            scientist = AIchemist()
            history = []
            
            for i in range(1, gens + 1):
                self.lbl_status.configure(text=f"System Status: CYCLE {i} - Screening")
                
                winner_data = run_screening_cycle(generation_id=i, allowed_elements=target_elements, logger=logger)
                
                if winner_data:
                    self.visualizer.start_simulation(winner_data['composition'])
                    
                    en = winner_data['predicted_energy']
                    unc = winner_data.get('uncertainty', 0.0)
                    comp_str = " ".join([f"{k}({v:.2f})" for k, v in winner_data['composition'].items()])
                    
                    self.update_metric_safe(self.metric_energy, f"{en:.4f} eV")
                    self.update_metric_safe(self.metric_uncertainty, f"±{unc:.4f}")
                    self.update_metric_safe(self.metric_composition, comp_str)
                    
                    history.append({
                        "generation": i,
                        "quantum_energy": winner_data['predicted_energy'],
                        "composition": winner_data['composition']
                    })
                    self.update_graph(history)
                
                self.lbl_status.configure(text=f"System Status: CYCLE {i} - Training")
                active_training_cycle(fine_tune_only=True)
                time.sleep(2)

            self.lbl_status.configure(text="System Status: GENERATING REPORT")
            if history:
                report = scientist.generate_campaign_summary(history)
                with open(os.path.join(logger.base_path, "Campaign_Report.md"), "w", encoding="utf-8") as f:
                    f.write(report)
            
            self.lbl_status.configure(text="System Status: COMPLETE")
            
        except Exception as e:
            print(f"Error: {e}")
            self.lbl_status.configure(text="System Status: ERROR")
        finally:
            self.btn_run.configure(state="normal", text="Start Campaign", fg_color="#27ae60")

if __name__ == "__main__":
    app = LaboratoryInterface()
    app.mainloop()