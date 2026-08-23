# AI-CFD-DiffusionUNet

> **Physics-Informed DiffusionUNet for Unsteady and Steady Airfoil Flowfield Prediction**  
> 基于物理感知扩散模型（PINN + DiffusionUNet）的翼型（NACA0012 / Eppler 387）非定常与定常流场快速重构与预测框架。

---

## 📌 项目亮点 (Highlights)

* **全尺寸 C-Grid 网格建模**：支持 $145 \times 689$ 高分辨率 C 型流场网格直接预测，精准捕获翼型前缘与边界层流场细观结构。
* **物理感知损失函数 (PINN Loss)**：在 Standard DPM 噪声损失的基础上，融合**边界层指数加权**、**连续性方程（散度）**、**涡度场匹配**以及**空间物理梯度算子**。
* **GPU 显存 Direct Batching**：优化 HPC 训练管线，支持全量流场张量驻留 GPU 显存与数据增强（镜像翻转与流速物理反转），实现零磁盘 IO 阻塞训练。
* **混合精度与极速训练**：集成 PyTorch AMP (`bfloat16`/`fp16`) 与 `GradScaler`，适配 NVIDIA A10/RTX 4060 等 GPU。

---

## 📂 目录结构 (Directory Structure)

```text
.
├── Steady FC/                 # 定常流场 DiffusionUNet 模块
│   └── DIffsionUnet/
│       └── Code/              # 训练、数据重采样与评估脚本
├── Unsteady/                  # 非定常流场 DiffusionUNet 核心模块
│   └── DIffsionUnet/
│       ├── Code/
│       │   ├── model_utils.py # DiffusionUNet 网络架构定义
│       │   ├── train_full.py  # 145x689 全尺寸物理感知训练主程序
│       │   ├── eval_full.py   # 模型测试与流场重构评估
│       │   ├── CFDLib.py      # CFD 数据解析与后处理工具库
│       │   └── Userinterface.py
│       └── Results/           # 存储 npz 数据集与 .pth 模型权重 (已 Git 忽略)
└── .gitignore                 # 大文件与日志过滤规则
