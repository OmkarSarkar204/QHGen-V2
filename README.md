# QHGen -: Quantum–Hybrid Materials Discovery Engine

## Overview

QHGen is a physics-informed computational framework designed to accelerate the discovery of efficient, low-cost electrocatalysts for hydrogen evolution reactions (HER). The system integrates machine learning, quantum-inspired modeling, and electrochemical theory to screen large compositional spaces while maintaining physical interpretability.

Unlike black-box prediction pipelines, QHGen explicitly incorporates thermodynamic constraints, electrochemical corrections, and material stability rules, enabling realistic and scientifically defensible predictions.

---

## Core Objective

To identify earth-abundant alloy compositions (e.g., Ni–Fe–Co systems) that exhibit near-optimal hydrogen adsorption energetics while remaining chemically stable under realistic electrochemical conditions.

---

## System Architecture

QHGen is structured as a modular, multi-stage pipeline:

### 1. Candidate Generation
- Compositional search using evolutionary strategies
- Supports constrained or free exploration of alloy space
- Generates surface slab structures using atomic substitution rules

### 2. Surrogate Energy Prediction
- Graph Neural Network (SchNet-based)
- Trained on atomistic configurations
- Predicts adsorption energies efficiently at scale

### 3. Physics-Based Corrections
To avoid unphysical predictions, the model applies deterministic corrections:

#### a. pH Correction (Nernst Approximation)
\[
\Delta G_{pH} = 0.059 \times \text{pH}
\]

Models proton chemical potential shifts under different electrochemical conditions.

#### b. Solvation Stabilization
Approximates solvent screening effects using a linear stabilization term proportional to exposed surface atoms.

These corrections allow the model to approximate electrochemical behavior without expensive explicit solvation.

### 4. Active Learning Loop
- Top candidates are validated using higher-fidelity evaluation
- Results are stored and used to retrain the surrogate model
- Improves prediction quality over successive generations

---

## Key Features

- Physics-informed ML (not purely data-driven)
- Modular architecture for rapid experimentation
- Scalable to large compositional spaces
- Compatible with future DFT or experimental validation
- Designed for transparency and reproducibility

---

## Scientific Scope and Limitations

### What This Model Does
- Approximates adsorption energetics under electrochemical conditions
- Identifies promising catalyst compositions
- Provides relative ranking and trends

### What This Model Does Not Do
- Perform explicit solvent molecular dynamics
- Solve full constant-potential DFT equations
- Model reaction kinetics or transition states

These simplifications are intentional to enable scalable screening.

---

## Recommended Usage

- Pre-screening catalyst compositions
- Guiding high-throughput DFT studies
- Exploring compositional trends
- Educational and research prototyping

---

## Future Extensions

- Integration of implicit solvent DFT data
- Multi-objective optimization (activity + stability)
- Uncertainty-aware surrogate modeling
- Coupling with experimental datasets

---

## Project Structure (Simplified)

