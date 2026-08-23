# AI-CFD-DiffusionUNet

> **Physics-Informed DiffusionUNet for Unsteady and Steady Airfoil Flowfield Prediction**  
> A deep learning framework based on Physics-Informed Diffusion Models (PINN + DiffusionUNet) for rapid reconstruction and prediction of steady and unsteady airfoil flowfields (NACA0012 / Eppler 387).

---

## 📌 Highlights

* **Full-Resolution C-Grid Modeling**: Direct predictions on $145 \times 689$ high-resolution C-type computational grids, accurately capturing leading-edge and boundary layer flow structures.
* **Physics-Informed Loss (PINN Loss)**: Integrates **Boundary Layer Exponential Weighting**, **Continuity Equation (Divergence)**, **Vorticity Matching**, and **Jacobian Spatial Gradient Operators** on top of standard DPM noise loss.
* **Direct GPU Memory Batching**: Optimized HPC training pipeline where full flowfield tensors reside directly in GPU VRAM alongside real-time data augmentation (horizontal flips and velocity physical inversion) for zero I/O latency.
* **Mixed Precision Acceleration**: Integrated PyTorch AMP (`bfloat16`/`fp16`) with `GradScaler` for speedups on NVIDIA A10 / RTX 4060 GPUs.

---

## 📂 Repository Structure

```text
.
├── .gitignore
├── Steady FC
│   └── DIffsionUnet
│       └── Code
│           ├── CFDLib.py
│           ├── highres_data_resample.py
│           ├── highres_evaluate.py
│           ├── highres_evaluate_fulltest.py
│           ├── highres_muti_eval.py
│           ├── model_utils.py
│           └── train_highres.py
├── Unsteady
│   └── DIffsionUnet
│       └── Code
│           ├── CFDLib.py
│           ├── Userinterface.py
│           ├── data_resample_full.py
│           ├── eval_full.py
│           ├── model_utils.py
│           ├── modeltest.py
│           └── train_full.py
<<<<<<< HEAD
└── ppython.py
=======
└── ppython.py
>>>>>>> c6525ac39f308e4f503f35bb2fa6a2f2db1ccbc9
