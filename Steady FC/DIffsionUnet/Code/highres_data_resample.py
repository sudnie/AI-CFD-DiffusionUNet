#usr/bin/python3
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# =================================================================
# 1. 基础配置与任务分配
# =================================================================
current_script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else "."
sim_data_path = os.path.abspath(os.path.join(current_script_dir, "../../../Database/Simdata_airfoil_Steady_FC/"))
res_data_path = os.path.abspath(os.path.join(current_script_dir, "../Results/"))
os.makedirs(res_data_path, exist_ok=True)

Mach_indices = np.array(['1', '2', '3', '4', '5', '6', '7', '8', '9'])
AoA_indices = np.array(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'])

ZONE_I = 401    # 原始 I 方向网格数
ZONE_J = 81     # 原始 J 方向网格数

# 物理边界限定区（保持你最饱满的近场剪裁比例，不进行任何 64x64 的压缩）
J_KEEP = 60      
I_START = 50     
I_END = 351      
# 裁剪后的目标尺寸：高 H = 60, 宽 W = (351 - 50) = 301

# 严格锁定 Table 4 对应的学术测试工况索引
target_test_cases = [(2, 9), (2, 14), (2, 5), (4, 9), (4, 14), (4, 5), (6, 9), (6, 14), (6, 5), (8, 9), (8, 14), (8, 5)]

file_tasks = []
for i_idx, i in enumerate(Mach_indices, start=1):
    for j_idx, j in enumerate(AoA_indices, start=1):
        filename = os.path.join(sim_data_path, f"flo_{i}_{j}.dat")
        if os.path.exists(filename):
            file_tasks.append({
                "path": filename, 
                "mach": 0.3 + (i_idx - 1) * 0.025, 
                "aoa": 0.0 + (j_idx - 1) * 0.5, 
                "indices": (i_idx, j_idx)
            })

# =================================================================
# 2. ⚡ 高保真原位裁剪引擎 (无重采样，保留纯净物理离散状态)
# =================================================================
def crop_highres_fluid_field(file_path):
    df = pd.read_csv(file_path, skiprows=2, header=None, sep=r'\s+', engine='c')
    raw_data = df.to_numpy()

    # 🎯 核心兜底：若部分工况缺少网格点（如 32480），边缘复制对齐到标准的 32481 (81*401)
    standard_total_nodes = ZONE_J * ZONE_I
    if raw_data.shape[0] != standard_total_nodes:
        raw_data = np.pad(
            raw_data[:standard_total_nodes], 
            ((0, max(0, standard_total_nodes - raw_data.shape[0])), (0, 0)), 
            mode='edge'
        )

    # 展开为 2D 连续拓扑物理面
    grid_x = raw_data[:, 0].reshape(ZONE_J, ZONE_I)
    grid_y = raw_data[:, 1].reshape(ZONE_J, ZONE_I)
    grid_fields = raw_data[:, [3, 4, 5, 7]].reshape(ZONE_J, ZONE_I, 4) # 提取 U, V, Rho, P

    # 实施原位精确剪裁，绝不调用任何插值，直接保留原始高分辨颗粒度
    cropped_x = grid_x[:J_KEEP, I_START:I_END]  
    cropped_y = grid_y[:J_KEEP, I_START:I_END]  
    cropped_fields = grid_fields[:J_KEEP, I_START:I_END, :] # 形状为 (60, 301, 4)

    # 调换维度使之符合 PyTorch 图像标准布局 [Channels, H, W] -> [4, 60, 301]
    cropped_fields_pytorch = cropped_fields.transpose(2, 0, 1)

    return cropped_fields_pytorch, cropped_x, cropped_y

# =================================================================
# 3. 核心分配循环与全量缓存
# =================================================================
Traindata, Testdata = [], []
Trainlabel_raw, Testlabel_raw = [], []
grid_x_cached, grid_y_cached = None, None

for task in tqdm(file_tasks, desc="🟩 [高精流场重构] 正在提取未缩放限定域矩阵"):
    cropped_fields, xc_g, yc_g = crop_highres_fluid_field(task["path"])
    
    if grid_x_cached is None:
        grid_x_cached, grid_y_cached = xc_g, yc_g

    if task["indices"] in target_test_cases:
        Testdata.append(cropped_fields)
        Testlabel_raw.append([task["mach"], task["aoa"]])
    else:
        Traindata.append(cropped_fields)
        Trainlabel_raw.append([task["mach"], task["aoa"]])

Traindata = np.array(Traindata)          
Testdata = np.array(Testdata)           
Trainlabel_raw = np.array(Trainlabel_raw)
Testlabel_raw = np.array(Testlabel_raw)

# =================================================================
# 4. 高保真全局归一化系数清算与落盘
# =================================================================
# 保持 4 通道独立最大/最小值的统计张量形态 [1, 4, 1, 1]
fields_min = np.zeros((1, 4, 1, 1))
fields_max = np.zeros((1, 4, 1, 1))

for c in range(4):
    fields_min[0, c, 0, 0] = Traindata[:, c, :, :].min()
    fields_max[0, c, 0, 0] = Traindata[:, c, :, :].max()

label_min = Trainlabel_raw.min(axis=0, keepdims=True)
label_max = Trainlabel_raw.max(axis=0, keepdims=True)

# 独立存储高分辨率专用的归一化因子
np.savez(os.path.join(res_data_path, "normalization_factors_highres.npz"), 
         fields_min=fields_min, fields_max=fields_max, label_min=label_min, label_max=label_max)

# 5. 执行数据单位空间映射与最终落盘
for mode in ['train', 'test']:
    fields = Traindata if mode == 'train' else Testdata
    labels = Trainlabel_raw if mode == 'train' else Testlabel_raw
    save_path = os.path.join(res_data_path, f"HighRes_airfoil_{mode}.npz")

    # 对流场特征执行严格的 [-1, 1] 正向区间压缩映射
    fields_norm = np.zeros_like(fields)
    for c in range(4):
        f_min_c = fields_min[0, c, 0, 0]
        f_max_c = fields_max[0, c, 0, 0]
        fields_norm[:, c, :, :] = 2.0 * (fields[:, c, :, :] - f_min_c) / (f_max_c - f_min_c + 1e-8) - 1.0
        
    # 对控制工况标签执行 [0, 1] 压缩映射
    labels_norm = (labels - label_min) / (label_max - label_min + 1e-8)                                                                                                                                                                                                                                           

    # 存储带有 (60, 301) 原始不规则高分辨率大坐标的全新训练就绪包
    np.savez(save_path, x=fields_norm, y=labels_norm, grid_x=grid_x_cached, grid_y=grid_y_cached)  
    print(f"🎉 {mode.capitalize()} 高清训练集无损转换成功 | 阵列维度: {fields_norm.shape} (H=60, W=301)")