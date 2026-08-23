#/usr/bin/python3
"""
C-grid 全尺寸物理数据预处理脚本
保留完整流场与网格（不进行裁剪与重采样）：
- 法向层数 J: 145 (0 ~ 144)
- 流向范围 I: 689 (0 ~ 688)
直接保存全尺寸张量 [4, 145, 689]。
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

current_script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

sim_data_path = os.path.abspath(os.path.join(current_script_dir, "../../../Database/Simdata_airfoil_unsteady/sol01_RANS3/"))
res_data_path = os.path.abspath(os.path.join(current_script_dir, "../Results/"))

if not os.path.exists(sim_data_path):
    sim_data_path = r"D:\gitHUBF\droneCV\CFD\Database\Simdata_airfoil_unsteady\sol01_RANS3"
    res_data_path = r"D:\gitHUBF\droneCV\CFD\AI-CFD-Technical-Report-main\Ch2. Unsteady\DIffsionUnet\Results"

print(f"📂 读取数据路径: {sim_data_path}")
print(f"📂 保存结果路径: {res_data_path}")
os.makedirs(res_data_path, exist_ok=True)

# ==========================================
# ⚙️ 预处理控制配置
# ==========================================
ZONE_I = 689    # 原始流向网格数
ZONE_J = 145    # 原始法向网格数
total_nodes = ZONE_I * ZONE_J

# 🌟 模式固定为 "full"：保留全尺寸网格
PREPROCESS_MODE = "full"

START_TIME = 100
END_TIME = 180

file_tasks = []
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

# ==========================================
# 全尺寸网格读取函数
# ==========================================
def process_snapshot(file_path, mode="full"):
    # 1. 读取原始数据
    df = pd.read_csv(file_path, skiprows=2, header=None, delimiter=r'\s+', engine='c')
    raw_data = np.nan_to_num(df.to_numpy())

    if raw_data.shape[0] != total_nodes:
        if raw_data.shape[0] > total_nodes:
            raw_data = raw_data[:total_nodes, :]
        else:
            raw_data = np.pad(raw_data, ((0, total_nodes - raw_data.shape[0]), (0, 0)), mode='edge')

    # 重构为原始 2D C-grid [145, 689]
    grid_x = raw_data[:, 0].reshape(ZONE_J, ZONE_I)
    grid_y = raw_data[:, 1].reshape(ZONE_J, ZONE_I)
    grid_fields = raw_data[:, [3, 4, 5, 7]].reshape(ZONE_J, ZONE_I, 4)  # [rho, u, v, p]

    if mode == "full":
        final_x = grid_x
        final_y = grid_y
        final_fields = grid_fields
    else:
        raise ValueError(f"未知的模式: {mode}")

    # 转置为通道优先布局: [4, H, W] -> [4, 145, 689]
    feat_tensor = np.transpose(final_fields, (2, 0, 1)) 
    return feat_tensor, final_x, final_y

# ==========================================
# 数据处理与划分 (80% Train, 20% Test 均匀交错采样)
# ==========================================
Traindata, Testdata = [], []
Trainlabel_raw, Testlabel_raw = [], []
grid_x_cached, grid_y_cached = None, None

desc_str = f"🟩 [全尺寸数据提取] 转换 {ZONE_J}x{ZONE_I} 物理张量"
for idx, task in enumerate(tqdm(file_tasks, desc=desc_str)):
    fields, xc_g, yc_g = process_snapshot(task["path"], mode=PREPROCESS_MODE)
    
    if grid_x_cached is None:
        grid_x_cached, grid_y_cached = xc_g, yc_g

    time_label = [task["time_step"]]

    # 每 5 帧提取 1 帧用于测试集
    if idx % 5 == 4:
        Testdata.append(fields)
        Testlabel_raw.append(time_label)
    else:
        Traindata.append(fields)
        Trainlabel_raw.append(time_label)

Traindata = np.array(Traindata)          # [65, 4, 145, 689]
Testdata = np.array(Testdata)            # [16, 4, 145, 689]
Trainlabel_raw = np.array(Trainlabel_raw)
Testlabel_raw = np.array(Testlabel_raw)

print(f"📊 划分结果: 训练集 {len(Traindata)} 帧 | 测试集 {len(Testdata)} 帧")

# ==========================================
# 保存归一化系数与 NPZ 文件
# ==========================================
fields_min = np.zeros((1, 4, 1, 1))
fields_max = np.zeros((1, 4, 1, 1))

for c in range(4):
    fields_min[0, c, 0, 0] = Traindata[:, c, :, :].min()
    fields_max[0, c, 0, 0] = Traindata[:, c, :, :].max()

label_min = Trainlabel_raw.min(axis=0, keepdims=True)
label_max = Trainlabel_raw.max(axis=0, keepdims=True)

suffix = "full"

np.savez(os.path.join(res_data_path, f"normalization_factors_{suffix}.npz"), 
         fields_min=fields_min, fields_max=fields_max, label_min=label_min, label_max=label_max)

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

    np.savez(save_path, x=fields_norm, y=labels_norm, grid_x=grid_x_cached, grid_y=grid_y_cached)  
    print(f"🎉 {mode.capitalize()} 数据集生成完成 | 张量形状: {fields_norm.shape}")