# 🌊 Physics-Informed Super-Resolution for Fluid Dynamics

## ⚡ Project Overview

This project demonstrates the use of Deep Learning to **accelerate Computational Fluid Dynamics (CFD)** simulations. We train a Neural Network to perform **Super-Resolution (SR)** on low-fidelity fluid velocity fields, bypassing the need for computationally expensive numerical solvers at high resolutions. The core innovation is integrating **Physics Constraints** into the training process.

## 🔄 Project Pipeline & Methodology

Our approach follows a strict pipeline from physics-based data generation to comparative model training:

### 1. Synthetic Data Generation

We utilize **PhiFlow** (a differentiable physics engine) to create ground-truth fluid simulations.

- **High-Resolution (Target):** $256 \times 256$ velocity fields ($u_x, u_y$).
- **Low-Resolution (Input):** $32 \times 32$ fields, created by **Average Pooling** the high-res simulation. This mimics sensor integration and ensures the input physically represents a coarse average of the fine grid.

### 2. Temporal Data Processing

To capture flow dynamics and temporal coherence, the model does not look at a single frame in isolation.

- **Input:** A tensor stack of **3 consecutive Low-Res frames** ($t-1$, $t$, $t+1$).
- **Output:** A single **High-Res frame** at time $t$.
  This allows the network to infer velocity direction and acceleration from the low-resolution context.

### 3. Model Architecture: Residual U-Net

We employ a **U-Net** architecture enhanced with **Residual Blocks**. This structure allows for deep feature extraction while preserving spatial information through skip connections, which is critical for resolving fine turbulent structures.

### 4. Experimental Comparison

We train two distinct variations of the model to quantify the impact of physics constraints:

1.  **Baseline Res-U-Net:** Trained purely on data loss (MSE between predicted and ground truth pixels).
2.  **Physics-Informed Res-U-Net:** Trained with a composite loss function: **MSE + Physics Loss**. The physics loss calculates the divergence of the generated field ($\nabla \cdot \mathbf{u}$) and penalizes non-zero values, enforcing the physical law of conservation of mass.

---

## 🎯 The Core Problem & Goal

| Problem                                                          | Goal                                                                                     |
| :--------------------------------------------------------------- | :--------------------------------------------------------------------------------------- |
| Traditional solvers take **seconds/minutes** per high-res frame. | Achieve **significant inference speedup** (milliseconds per frame) via a neural network. |
| Standard AI (MSE) outputs **unphysical, blurry** flow.           | Enforce **Navier-Stokes** laws to ensure outputs are physically valid and sharp.         |

## 🚀 Get Started

1.  Clone the repository: `git clone [REPO_URL]`
2.  Install dependencies: `pip install -r requirements.txt`
3.  Run the data generator: `cd src & python generate_dataset.py`

<br/>
<div align="center">
  <img src="data/sim_00_jet.gif" width="512" height="256" alt="High-Res vs Low-Res Simulation" />
  <p><em>Left: High-Resolution Ground Truth | Right: Low-Resolution Input</em></p>
</div>
