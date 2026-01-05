# 🌊 Physics-Informed Super-Resolution for Fluid Dynamics

**Bridging the gap between coarse simulations and high-fidelity reality using Deep Learning.**

## 🚀 Project Overview and Goal
Simulating high-resolution fluid dynamics is computationally expensive. This project explores a **Deep Learning approach to Super-Resolution (SR)**, enabling the reconstruction of fine-grained turbulent details from low-resolution, "blocky" inputs.

**Our Goal:**
To develop a model that doesn't just "upscale" images, but **understands physics**. By integrating **Navier-Stokes constraints** directly into the loss function, we ensure that our generated high-resolution fields ($256 \times 256$) are not only visually sharp but physically valid divergence-free flows.

---

## 🛠️ Project Pipeline & Methodology

Our pipeline treats fluid simulation as a "Sim-to-Real" problem. We generate data using the **PhiFlow** toolkit, creating pairs of **Low-Res ($64^2$)** inputs and **High-Res ($256^2$)** ground truth targets across 4 physical channels: *Velocity (u, v), Pressure (p), and Smoke Density (s).*

We investigated two distinct architectural approaches to solve this challenge:

### 1. ResUNet
* **Input:** Stack of 3 consecutive frames `[t-1, t, t+1]` to capture temporal flow dynamics.
* **Architecture:** A **Residual U-Net** with deep encoder-decoder paths and skip connections.
* **Focus:** Smoothness and temporal consistency. The model takes pre-upsampled coarse inputs and refines them to match the target.

### 2. SRGAN
* **Input:** Single-frame physics fields (4 channels).
* **Architecture:** A **Super-Resolution GAN** composed of:
    * **Generator:** Modified SRResNet (16 Residual Blocks, PReLU) for 4x spatial upsampling.
    * **Discriminator:** VGG-style network to differentiate real vs. generated turbulence.
* **Physics-Informed Training:**
    * **Phase 1 (Pre-training):** Supervised learning on mathematically downscaled inputs.
    * **Phase 2 (Fine-tuning):** Finetuning with a dataset where inputs are generated natively at low resolution (64x64)


<br/>
<div align="center">
  <img src="data/sim_00_jet.gif" width="512" height="256" alt="High-Res vs Low-Res Simulation" />
  <p><em>Left: High-Resolution Ground Truth | Right: Low-Resolution Input</em></p>
</div>
