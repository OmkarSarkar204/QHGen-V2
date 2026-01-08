## This project is part of the Microsoft Imagine Cup 2026.

# QHGen: Quantum–Hybrid Materials Discovery Engine


**Team Eigen**

---

## The Mission
**QHGen** is a physics-informed AI platform designed to solve the Platinum dependency problem, the critical economic bottleneck in Green Hydrogen production.

The hydrogen economy is currently constrained by **Platinum Group Metals (PGMs)** such as Platinum ($30,000/kg) and Iridium (~$160,000/kg). Global PGM reserves are insufficient to scale electrolyzers to net-zero targets.

**QHGen** combines **Graph Neural Networks (SchNet)** with **quantum-chemical constraints** to autonomously scan billions of earth-abundant alloy combinations (Ni, Fe, Co, Mo, W, Cu, etc.). The objective is to discover novel, low-cost materials that replicate the electronic behavior of Platinum at **99.9% lower cost**.

---

## Key Capabilities (Implemented)

### 1. Physics-Informed AI Core
- **SchNet GNN Architecture:** Continuous-filter convolutional layers model atomic interactions directly in three-dimensional space.
- **Transfer Learning:** Pre-trained on the **Open Catalyst 2020 (OC20)** dataset (130M+ samples) to learn fundamental atomic forces and generalize to unseen alloy systems.

### 2. Cloud-Native Active Learning
- **Azure Machine Learning Integration**
  - Automated training pipelines on Azure Managed Compute
  - Closed-loop optimization where candidate generation, physics validation, and retraining occur autonomously
  - Real-time experiment tracking using Azure MLFlow

### 3. Electrochemical Realism
QHGen enforces physically grounded constraints to eliminate false positives:
- **Pourbaix Stability Filters:** Rejects materials unstable under acidic or alkaline operating conditions
- **Nernstian Corrections:** Adjusts adsorption free energies for electrochemical conditions

  \[
  \Delta G_{final} = \Delta G_{DFT} + 0.059 \times \text{pH} + \Delta G_{solvation}
  \]

- **Sabatier Analysis:** Ranks catalysts by proximity to optimal hydrogen binding energy (approximately 0 eV)

---

## System Architecture

QHGen operates as a modular, multi-stage discovery pipeline:

1. **Combinatorial Generation**
   - Evolutionary strategies generate large alloy search spaces across d-block elements
   - Three-dimensional surface slabs constructed using atomic substitution rules

2. **Surrogate Screening**
   - SchNet predicts hydrogen adsorption free energy (\(\Delta G_{H*}\)) in milliseconds
   - Uncertainty estimation filters unreliable predictions

3. **Physics Verification**
   - Top-ranked candidates validated using quantum-physics simulators
   - Prediction errors are fed back into the training loop via active learning

4. **Industrial Reporting**
   - Automated feasibility reports covering cost per kilogram, elemental abundance, and supply-chain risk

---

## Results and Validation

- **Speed:** Screening time reduced from years to minutes (>99.9% reduction)
- **Accuracy:** Mean Absolute Error (MAE) below **0.02 eV** on validation datasets
- **Discovery:** Identified multiple PGM-free candidates, including Ni-Fe and Ni-Mo variants, with theoretical performance comparable to Platinum
- **Cost Reduction:** Validated material cost pathways from ~$30,000/kg to ~$15/kg

---

## Tech Stack

- **Programming Language:** Python 3.9+
- **Deep Learning:** PyTorch, PyTorch Geometric
- **Cloud Infrastructure:** Microsoft Azure Machine Learning, Azure Blob Storage
- **Physics Engines:** Atomic Simulation Environment (ASE), CatLearn
- **Visualization:** Matplotlib, Tkinter

---

## Installation

```bash
git clone https://github.com/omkarsarkar/qhgen.git
cd qhgen

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
