# 🌊 Physics-Informed Super-Resolution for Fluid Dynamics

## ⚡ Project Overview

This project demonstrates the use of Deep Learning to **accelerate Computational Fluid Dynamics (CFD)** simulations. We train a Neural Network to perform **Super-Resolution (SR)** on low-fidelity fluid velocity fields, bypassing the need for computationally expensive numerical solvers at high resolutions. The core innovation is integrating **Physics Constraints** into the training process.

## 🎯 The Core Problem & Goal

| Problem                                                          | Goal                                                                                     |
| :--------------------------------------------------------------- | :--------------------------------------------------------------------------------------- |
| Traditional solvers take **seconds/minutes** per high-res frame. | Achieve **significant inference speedup** (milliseconds per frame) via a neural network. |
| Standard AI (MSE) outputs **unphysical, blurry** flow.           | Enforce **Navier-Stokes** laws to ensure outputs are physically valid and sharp.         |

## 🛠️ Technology Stack

- **Framework:** PyTorch
- **Physics Engine:** PhiFlow (Differentiable)
- **Data Format:** NumPy (.npy)
- **Core Method:** Physics-Informed Super-Resolution (PINN-SR) with U-Net architecture.

## 🧠 Key Innovations (The "Why")

1.  **Divergence Minimization:** The model is penalized if the generated velocity field violates the **Conservation of Mass** (i.e., the divergence of the flow is not zero).
2.  **Synthetic Data Generation:** A robust, **64-bit precision** pipeline generates large datasets of paired Low-Resolution ($32 \times 32$) and High-Resolution ($256 \times 256$) fluid frames.

## 🚀 Get Started

1.  Clone the repository: `git clone [REPO_URL]`
2.  Install dependencies: `pip install -r requirements.txt`
3.  Run the data generator: `cd src & python generate_dataset.py`

|                                       Input (Low Res 32x32)                                       |                   Target (High Res 128x128)                   |
| :-----------------------------------------------------------------------------------------------: | :-----------------------------------------------------------: |
| <img src="data/simulation_lr.gif" width="256" height="256" style="image-rendering: pixelated;" /> | <img src="data/simulation_hr.gif" width="256" height="256" /> |

_(Further instructions to be added in the full version.)_
