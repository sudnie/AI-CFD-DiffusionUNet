#/usr/bin/python3
"""
核心区 (120x669) 物理流场可视化脚本
读取 Diffusion_airfoil_unsteady_core_train.npz 并在物理 C-grid 上渲染真实流场云图
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# ⚙️ 可视化配置
# ==========================================
current_script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
res_data_path = os.path.abspath(os.path.join(current_script_dir, "../Results/"))

# 🌟 对应刚才生成的核心区后缀
DATA_SUFFIX = "core"    # 读取 Diffusion_airfoil_unsteady_core_train.npz
SPLIT = "train"        # "train" 或 "test"
SAMPLE_IDX = 0          # 要查看的物理快照索引

# 文件路径解析
npz_file_path = os.path.join(res_data_path, f"Diffusion_airfoil_unsteady_{DATA_SUFFIX}_{SPLIT}.npz")
norm_file_path = os.path.join(res_data_path, f"normalization_factors_{DATA_SUFFIX}.npz")

if not os.path.exists(npz_file_path):
    raise FileNotFoundError(f"❌ 找不到核心区数据集文件: {npz_file_path}")

# ==========================================
# 1. 读取数据与反归一化恢复物理量
# ==========================================
print(f"📦 读取核心区数据集: {npz_file_path}")
data = np.load(npz_file_path)
norm_factors = np.load(norm_file_path)

x_norm = data['x']        # [N, 4, 120, 669]
y_norm = data['y']        # [N, 1]
grid_x = data['grid_x']   # [120, 669]
grid_y = data['grid_y']   # [120, 669]

f_min = norm_factors['fields_min']  # [1, 4, 1, 1]
f_max = norm_factors['fields_max']  # [1, 4, 1, 1]
l_min = norm_factors['label_min']
l_max = norm_factors['label_max']

# 提取指定帧数据
norm_sample = x_norm[SAMPLE_IDX]
time_val = (y_norm[SAMPLE_IDX] * (l_max - l_min + 1e-8) + l_min).item()

# 反归一化物理量恢复
phys_0_1 = (norm_sample + 1.0) / 2.0
phys_sample = phys_0_1 * (f_max[0] - f_min[0]) + f_min[0] # [4, 120, 669]

rho = phys_sample[0]  # 密度
u   = phys_sample[1]  # U 速度
v   = phys_sample[2]  # V 速度
p   = phys_sample[3]  # 压力

print(f"✅ 成功读取物理帧 #{SAMPLE_IDX} | 时间步: t = {time_val:.1f}")
print(f"📐 核心区网格节点维度: {grid_x.shape[0]} × {grid_x.shape[1]}")

# ==========================================
# 2. 绘制 4 通道核心区真实物理场云图
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 8))
fig.suptitle(f"Core Area Physical Field (120x669) | Snapshot #{SAMPLE_IDX} (t = {time_val:.1f})", fontsize=15, fontweight='bold')

fields_info = [
    ("Density (ρ)", rho, "kg/m³", axes[0, 0]),
    ("U Velocity", u, "m/s", axes[0, 1]),
    ("V Velocity", v, "m/s", axes[1, 0]),
    ("Pressure (P)", p, "Pa", axes[1, 1])
]

cmap = 'jet'

for title, field_data, unit, ax in fields_info:
    # 🌟 pcolormesh 在 C-grid 上实现完全平滑连续色彩过渡
    mesh = ax.pcolormesh(grid_x, grid_y, field_data, shading='gouraud', cmap=cmap)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(unit, fontsize=10)
    
    # 绘制 J=0 (翼型近壁面/壁面轮廓线)
    ax.plot(grid_x[0, :], grid_y[0, :], 'k-', linewidth=0.8, alpha=0.7)
    
    ax.set_title(f"{title} | Range: [{field_data.min():.3f}, {field_data.max():.3f}]", fontsize=11, fontweight='bold')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    
    # 设置合理的核心区观察视角范围 (针对翼型和近尾迹区)
    ax.set_xlim([grid_x.min(), grid_x.max()])
    ax.set_ylim([grid_y.min(), grid_y.max()])
    ax.set_aspect('equal', adjustable='box')

plt.tight_layout()

# 保存渲染结果
save_plot_path = os.path.join(res_data_path, f"preprocessed_field_{DATA_SUFFIX}_frame{SAMPLE_IDX}.png")
plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
print(f"🖼️ 核心区流场渲染图已保存至: {save_plot_path}")
plt.show()