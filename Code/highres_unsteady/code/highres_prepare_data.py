#!/usr/bin/python3#-*- coding: utf-8 -*-
"""
C-grid 全尺寸物理数据预处理脚本（支持下采样至 1/4 分辨率）

- 原始法向层数 J: 145 → 下采样后 36
- 原始流向范围 I: 689 → 下采样后 172
- 最终张量形状: [4, 36, 172]
- 使用平均池化，裁剪至4的整数倍后取块平均
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

#==========================================#⚙️ 预处理控制配置#==========================================ZONE_I = 689          # 原始流向网格数
ZONE_J = 145          # 原始法向网格数
total_nodes = ZONE_I * ZONE_J

DOWNSAMPLE_FACTOR = 8   # 下采样倍率（边长缩小至 1/4）
PREPROCESS_MODE = "full"  # 保持原始全尺寸后再下采样

START_TIME = 100
END_TIME = 180

#路径设定（保持与原始脚本一致）current_script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
sim_data_path = os.path.abspath(os.path.join(current_script_dir, "../../../Database/Simdata_airfoil_unsteady/sol01_RANS3/"))
res_data_path = os.path.abspath(os.path.join(current_script_dir, "../Results/"))

if not os.path.exists(sim_data_path):
    sim_data_path = r"D:\gitHUBF\droneCV\CFD\Database\Simdata_airfoil_unsteady\sol01_RANS3"
    res_data_path = r"D:\gitHUBF\droneCV\CFD\AI-CFD-Technical-Report-main\Ch2. Unsteady\DIffsionUnet\Results"

print(f"📂 读取数据路径: {sim_data_path}")
print(f"📂 保存结果路径: {res_data_path}")
os.makedirs(res_data_path, exist_ok=True)

#==========================================#下采样函数（平均池化，裁剪至 factor 的整数倍）#==========================================def downsample(data, factor=4):
    """
    对 (..., J, I) 形状的数据进行平均池化下采样。
    仅支持最后两维为空间维度。
    """
    if factor <= 1:
        return data
    #确保维度顺序为 (..., J, I)    shape = data.shape
    J, I = shape[-2], shape[-1]
    J_new = J // factor
    I_new = I // factor
    #裁剪至 factor 的整数倍    data_crop = data[..., :J_new * factor, :I_new * factor]
    #重塑为 (..., J_new, factor, I_new, factor)    new_shape = list(shape[:-2]) + [J_new, factor, I_new, factor]
    data_reshaped = data_crop.reshape(new_shape)
    #对 factor 维度求平均 (最后两维的因子)    downsampled = data_reshaped.mean(axis=(-1, -3))   # 注意轴顺序：-1是I_factor, -3是J_factor
    return downsampled

#==========================================#全尺寸网格读取函数（含下采样）#==========================================def process_snapshot(file_path, mode="full", downsample_factor=4):
    #1. 读取原始数据    df = pd.read_csv(file_path, skiprows=2, header=None, delimiter=r'\s+', engine='c')
    raw_data = np.nan_to_num(df.to_numpy())

    if raw_data.shape[0] != total_nodes:
        if raw_data.shape[0] > total_nodes:
            raw_data = raw_data[:total_nodes, :]
        else:
            raw_data = np.pad(raw_data, ((0, total_nodes - raw_data.shape[0]), (0, 0)), mode='edge')

    #重构为原始 2D C-grid [145, 689]    grid_x = raw_data[:, 0].reshape(ZONE_J, ZONE_I)
    grid_y = raw_data[:, 1].reshape(ZONE_J, ZONE_I)
    grid_fields = raw_data[:, [3, 4, 5, 7]].reshape(ZONE_J, ZONE_I, 4)  # [rho, u, v, p]

    if mode == "full":
        final_x = grid_x
        final_y = grid_y
        final_fields = grid_fields
    else:
        raise ValueError(f"未知的模式: {mode}")

    #下采样（如果因子 > 1）    if downsample_factor > 1:
        #场量形状 [J, I, C] → 转置为 [C, J, I] 以便下采样        fields_t = np.transpose(final_fields, (2, 0, 1))  # [4, J, I]
        fields_down = downsample(fields_t, factor=downsample_factor)
        #网格坐标 [J, I]        grid_x_down = downsample(final_x, factor=downsample_factor)
        grid_y_down = downsample(final_y, factor=downsample_factor)
        #转回通道优先的张量（已经是 [C, J_new, I_new]）        feat_tensor = fields_down
        return feat_tensor, grid_x_down, grid_y_down
    else:
        #无下采样，保持通道优先        feat_tensor = np.transpose(final_fields, (2, 0, 1))  # [4, 145, 689]
        return feat_tensor, final_x, final_y

#==========================================#构建文件列表#==========================================file_tasks = []
for t in range(START_TIME, END_TIME + 1):
    time_str = str(t).rjust(3, '0')
    filename = os.path.join(sim_data_path, f"flo001.0000{time_str}uns")
    if os.path.exists(filename):
        file_tasks.append({
            "path": filename,
            "time_step": float(t),
            "filename": f"flo001.0000{time_str}uns"
        })

print(f"✅ 成功匹配到 {len(file_tasks)} 帧物理流场快照 (t = {START_TIME} ~ {END_TIME})")

#==========================================#数据处理与划分 (80% Train, 20% Test 均匀交错采样)#==========================================Traindata, Testdata = [], []
Trainlabel_raw, Testlabel_raw = [], []
grid_x_cached, grid_y_cached = None, None

desc_str = f"🟩 [下采样{DOWNSAMPLE_FACTOR}倍] 转换 {ZONE_J}x{ZONE_I} → 新尺寸"
for idx, task in enumerate(tqdm(file_tasks, desc=desc_str)):
    fields, xc_g, yc_g = process_snapshot(task["path"], mode=PREPROCESS_MODE,
                                          downsample_factor=DOWNSAMPLE_FACTOR)

    if grid_x_cached is None:
        grid_x_cached, grid_y_cached = xc_g, yc_g

    time_label = [task["time_step"]]

    #每 5 帧提取 1 帧用于测试集    if idx % 5 == 4:
        Testdata.append(fields)
        Testlabel_raw.append(time_label)
    else:
        Traindata.append(fields)
        Trainlabel_raw.append(time_label)

Traindata = np.array(Traindata)          # [N_train, 4, J_new, I_new]
Testdata = np.array(Testdata)            # [N_test,  4, J_new, I_new]
Trainlabel_raw = np.array(Trainlabel_raw)
Testlabel_raw = np.array(Testlabel_raw)

print(f"📊 划分结果: 训练集 {len(Traindata)} 帧 | 测试集 {len(Testdata)} 帧")
print(f"📐 下采样后张量形状: {Traindata.shape[1:]}")

#==========================================#保存归一化系数与 NPZ 파일#==========================================fields_min = np.zeros((1, 4, 1, 1))
fields_max = np.zeros((1, 4, 1, 1))

for c in range(4):
    fields_min[0, c, 0, 0] = Traindata[:, c, :, :].min()
    fields_max[0, c, 0, 0] = Traindata[:, c, :, :].max()

label_min = Trainlabel_raw.min(axis=0, keepdims=True)
label_max = Trainlabel_raw.max(axis=0, keepdims=True)

suffix = f"down{DOWNSAMPLE_FACTOR}"

np.savez(os.path.join(res_data_path, f"normalization_factors_{suffix}.npz"),
         fields_min=fields_min, fields_max=fields_max,
         label_min=label_min, label_max=label_max)

for mode in ['train', 'test']:
    fields = Traindata if mode == 'train' else Testdata
    labels = Trainlabel_raw if mode == 'train' else Testlabel_raw
    save_path = os.path.join(res_data_path, f"Diffusion_airfoil_unsteady_{suffix}_{mode}.npz")

    fields_norm = np.zeros_like(fields)
    for c in range(4):
        f_min_c = fields_min[0, c, 0, 0]
        f_max_c = fields_max[0, c, 0, 0]
        fields_norm[:, c, :, :] = 2.0 * (fields[:, c, :, :] - f_min_c) / (f_max_c - f_min_c + 1e-8) - 1.0

    labels_norm = (labels - label_min) / (label_max - label_min + 1e-8)

    np.savez(save_path,
             x=fields_norm,
             y=labels_norm,
             grid_x=grid_x_cached,
             grid_y=grid_y_cached)

    print(f"🎉 {mode.capitalize()} 数据集生成完成 | 张量形状: {fields_norm.shape}")