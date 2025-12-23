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

### 3. Model Architecture: SRGAN

We adapted the **Super-Resolution <generative Adversarial Network** to reconstruct high-fidelity fluid simulations from low-resolution (64x64) inputs. 

### 4. Training Strategy

1.  **Pre-training:** We first train the Generator in isolation
using Mean Squared Error (MSE) loss. This phase ini-
tializes the upscaling filters and stabilizes the model be-
fore introducing adversarial complexity.
2.  **Fine-tuning:** We then train the full GAN (Generator +
Discriminator) using a composite loss function. This stage introduces the **adversarial loss**  to recover high-frequency turbulent details and a **physics-informed loss**  to enforce fluid constraints.

---

## 🎯 The Core Problem & Goal

| Problem                                                          | Goal                                                                                     |
| :--------------------------------------------------------------- | :--------------------------------------------------------------------------------------- |
| Traditional solvers take **seconds/minutes** per high-res frame. | Achieve **significant inference speedup** (milliseconds per frame) via a neural network. |
| Standard AI (MSE) outputs **unphysical, blurry** flow.           | Enforce **Navier-Stokes** laws to ensure outputs are physically valid and sharp.         |

## 🚀 Get Started

1.  Clone the repository: `git clone https://github.com/Daniel200273/Physics-Informed-Flow-SR`
2.  Install dependencies: `pip install -r requirements.txt`
3.  Run the data generator: `cd src & python generate_dataset.py`

<br/>
<div align="center">
  <img src="data/sim_00_jet.gif" width="512" height="256" alt="High-Res vs Low-Res Simulation" />
  <p><em>Left: High-Resolution Ground Truth | Right: Low-Resolution Input</em></p>
</div>
