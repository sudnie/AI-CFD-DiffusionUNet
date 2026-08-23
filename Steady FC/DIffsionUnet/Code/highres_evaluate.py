#/usr/bin/python3
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 🌟 自适应高保真长方形去噪网络
from model_utils import HighResDiffusionUNet

# ==========================================
# 🌟 基于 x0-prediction 的正统反向采样器（高保真变尺寸适配）
# ==========================================
@torch.no_grad()
def sample_flow_field_scheme_b(model, device, grid_x, grid_y, target_cond, timesteps=1000):
    model.eval()
    
    # 严格对齐训练时的线性调度
    beta = torch.linspace(1e-4, 0.02, timesteps).to(device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    
    # 🎯 动态自适应当前传入的大网格 H, W，直接在原生高分辨率空间里注入高斯噪声
    h, w = grid_x.shape[1], grid_x.shape[2]
    x_t = torch.randn((1, 4, h, w), device=device)
    
    pbar = tqdm(reversed(range(timesteps)), desc="🌪️ 正在执行方案 B ($x_0$预测) 反向降噪演进", total=timesteps)
    for t_idx in pbar:
        t = torch.full((1,), t_idx, device=device, dtype=torch.long)
        
        # 1. 此时模型吐出的就是预估的无噪干净流场 x0_pred
        x0_pred = model(x_t, grid_x, grid_y, t, target_cond)
        
        # 2. 提取当前步的调度常数
        ab_t = alpha_bar[t_idx]
        
        if t_idx > 0:
            # 3. 当 t > 0 时，通过标准的 x0 预测公式推导 x_{t-1} 的均值部分
            ab_t_prev = alpha_bar[t_idx - 1]
            
            # 计算无噪预测项和当前噪声保留项的权重系数
            weight_x0 = torch.sqrt(ab_t_prev) * beta[t_idx] / (1.0 - ab_t)
            weight_xt = torch.sqrt(alpha[t_idx]) * (1.0 - ab_t_prev) / (1.0 - ab_t)
            
            mean = weight_x0 * x0_pred + weight_xt * x_t
            
            # 引入后验方差随机项
            var = (1.0 - ab_t_prev) / (1.0 - ab_t) * beta[t_idx]
            noise = torch.randn_like(x_t)
            
            x_t = mean + torch.sqrt(var) * noise
        else:
            # 4. 最后一步 (t=0)
            x_t = x0_pred
            
    return x_t

# ==========================================
# 🌟 与 CFDLib 绝对等价的单通道相对误差算子
# ==========================================
def relative_error_calc(pred, true, eps=1e-8):
    """
    计算展平后单通道的一阶离散相对误差
    """
    return np.sum(np.abs(pred - true)) / (np.sum(np.abs(true)) + eps)


# ==========================================
# 2. 自动化检索与评估主程序
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动高保真变尺寸方案B 专用流场评估引擎: {device}")
    
    # 🌟【测试案例选择】输入你想看的结果工况
    TARGET_MACH = 0.475
    TARGET_AOA = 6
    
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else "."
    # 🎯 对齐高保真专用的权重文件与参考数据集路径
    weights_path = os.path.abspath(os.path.join(current_dir, "../Results/airfoil_diffusion_highres_ep5000.pth"))
    norm_path = os.path.abspath(os.path.join(current_dir, "../Results/normalization_factors_highres.npz"))
    
    train_data_path = os.path.abspath(os.path.join(current_dir, "../Results/HighRes_airfoil_train.npz"))
    test_data_path = os.path.abspath(os.path.join(current_dir, "../Results/HighRes_airfoil_test.npz"))
    
    # 🎯 引入自适应高精度架构网络
    model = HighResDiffusionUNet().to(device)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("✅ 高保真物理模型权重载入完毕。")
    else:
        raise FileNotFoundError(f"❌ 找不到模型权重文件: {weights_path}，请先确保模型已被成功训练！")
    
    train_data = np.load(train_data_path)
    test_data = np.load(test_data_path)
    norm_factors = np.load(norm_path)
    
    f_min, f_max = norm_factors['fields_min'], norm_factors['fields_max']
    l_min, l_max = norm_factors['label_min'], norm_factors['label_max']
    grid_x_raw = train_data['grid_x'] 
    grid_y_raw = train_data['grid_y']
    
    # 全局最近邻检索
    target_phys = np.array([TARGET_MACH, TARGET_AOA])
    target_norm = (target_phys - l_min[0]) / (l_max[0] - l_min[0] + 1e-8)
    
    dists_in_train = np.linalg.norm(train_data['y'] - target_norm, axis=1)
    dists_in_test = np.linalg.norm(test_data['y'] - target_norm, axis=1)
    
    min_dist_train, idx_train = np.min(dists_in_train), np.argmin(dists_in_train)
    min_dist_test, idx_test = np.min(dists_in_test), np.argmin(dists_in_test)
    
    if min_dist_train <= min_dist_test:
        data_source = "Train Dataset"
        idx = idx_train
        matched_labels_norm = train_data['y'][idx]
        gt_fields_norm = train_data['x'][idx]
    else:
        data_source = "Test Dataset"
        idx = idx_test
        matched_labels_norm = test_data['y'][idx]
        gt_fields_norm = test_data['x'][idx]
        
    cond_phys = matched_labels_norm * (l_max[0] - l_min[0] + 1e-8) + l_min[0]
    print(f"🔍 真正场源匹配成功: 【{data_source}】 #内部行索引 {idx}")
    print(f"📊 [物理对齐验证] 最终抓取到的气动工况为：")
    print(f"    -> 真实世界马赫数 Mach: {cond_phys[0]:.4f} (期望: {TARGET_MACH:.4f})")
    print(f"    -> 真实世界攻角 AoA:    {cond_phys[1]:.4f}° (期望: {TARGET_AOA:.1f}°)")
    
    # 格式化输入特征
    cond_tensor = torch.tensor(target_norm, dtype=torch.float32, device=device).unsqueeze(0)
    grid_x_tensor = torch.tensor(grid_x_raw, dtype=torch.float32, device=device).unsqueeze(0)
    grid_y_tensor = torch.tensor(grid_y_raw, dtype=torch.float32, device=device).unsqueeze(0)
    
    # ⏱️ 高精计时启动
    start_wall_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_cuda_time = time.time()

    # 🎯 移除固定的 img_size 限制，由内部算子自适应大网格生成
    pred_field_norm = sample_flow_field_scheme_b(model, device, grid_x_tensor, grid_y_tensor, cond_tensor)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_time = time.time() - start_cuda_time
    print(f"⏱️ 采样计算耗时: {elapsed_time:.2f} 秒")

    # 反归一化
    def denormalize(field_tensor):
        arr = field_tensor.cpu().numpy()
        if len(arr.shape) == 4: arr = arr[0]
        phys_0_1 = (arr + 1.0) / 2.0
        f_min_c = f_min[0, :, 0, 0].reshape(4, 1, 1)
        f_max_c = f_max[0, :, 0, 0].reshape(4, 1, 1)
        return phys_0_1 * (f_max_c - f_min_c) + f_min_c
    
    gt_phys = denormalize(torch.tensor(gt_fields_norm))
    pred_phys = denormalize(pred_field_norm)
    err_phys = np.abs(gt_phys - pred_phys)
    
    # =================================================================
    # 🌟 核心融合重构区：双重体系误差全清算 🌟
    # =================================================================
    eps = 1e-5
    ch_u, ch_v, ch_p = 1, 2, 3 # 1-U速度, 2-V速度, 3-Pressure压力
    
    # ----- 体系一：传统的单通道离散相对误差 (对应 relative_error) -----
    error_u_relative = relative_error_calc(pred_phys[ch_u].flatten(), gt_phys[ch_u].flatten(), eps)
    error_v_relative = relative_error_calc(pred_phys[ch_v].flatten(), gt_phys[ch_v].flatten(), eps)
    error_p_relative = relative_error_calc(pred_phys[ch_p].flatten(), gt_phys[ch_p].flatten(), eps)
    
    # ----- 体系二：学术界常用的标准通道相对 L2 误差 -----
    l2_error_u = np.linalg.norm(gt_phys[ch_u] - pred_phys[ch_u]) / (np.linalg.norm(gt_phys[ch_u]) + eps)
    l2_error_p = np.linalg.norm(gt_phys[ch_p] - pred_phys[ch_p]) / (np.linalg.norm(gt_phys[ch_p]) + eps)
    
    # ----- 体系三：【Table 4 核心】全通道融合空间相对 L2 误差百分比 -----
    u_pred_f, u_true_f = pred_phys[ch_u].flatten(), gt_phys[ch_u].flatten()
    v_pred_f, v_true_f = pred_phys[ch_v].flatten(), gt_phys[ch_v].flatten()
    p_pred_f, p_true_f = pred_phys[ch_p].flatten(), gt_phys[ch_p].flatten()
    
    res_sq = np.sum((u_pred_f - u_true_f)**2) + np.sum((v_pred_f - v_true_f)**2) + np.sum((p_pred_f - p_true_f)**2)
    den_sq = np.sum(u_true_f**2) + np.sum(v_true_f**2) + np.sum(p_true_f**2)
    case_combined_l2_error = np.sqrt(res_sq / den_sq) * 100

    # 定位极值最高误差坐标点
    max_err_idx_u = np.unravel_index(np.argmax(err_phys[ch_u]), err_phys[ch_u].shape)
    max_err_idx_p = np.unravel_index(np.argmax(err_phys[ch_p]), err_phys[ch_p].shape)
    max_err_x_u, max_err_y_u = grid_x_raw[max_err_idx_u], grid_y_raw[max_err_idx_u]
    max_err_x_p, max_err_y_p = grid_x_raw[max_err_idx_p], grid_y_raw[max_err_idx_p]

    # 控制台全面清算打印
    print("\n📈 ========================================================")
    print(f"📊 [Diffusion 模型全气动通道多维误差报告]")
    print(f"    >> 备份单通道离散相对误差 (relative_error 映射):")
    print(f"       -> Error u: {error_u_relative:.6e}")
    print(f"       -> Error v: {error_v_relative:.6e}")
    print(f"       -> Error p: {error_p_relative:.6e}")
    print(f"    >> 单通道标准相对 L2 误差:")
    print(f"       -> U-Velocity 相对 L2 误差: {l2_error_u * 100:.3f}%")
    print(f"       -> Pressure   相对 L2 误差: {l2_error_p * 100:.3f}%")
    print(f"    -------------------------------------------------------")
    print(f"    >> 【Table 4 核心】全通道融合空间相对 L2 误差: {case_combined_l2_error:.4f}%")
    print("===========================================================\n")

    # ==========================================
    # 3. 稳态流场 3x2 标准可视化输出 (连续色彩升级版)
    # ==========================================
    print("🎨 正在渲染高保真平滑流场与误差捕获图谱...")
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f"High-Res Original Mesh Verification Matrix (Table 4 L2: {case_combined_l2_error:.3f}%)\nMach={TARGET_MACH:.3f}, AoA={TARGET_AOA:.1f}°", fontsize=13, fontweight='bold')
    
    cmap, err_cmap = 'jet', 'magma'
    
    # 🌟 将所有的 contourf 替换为 pcolormesh(..., shading='gouraud') 实现连续流畅过渡
    # --- 行 1：U 速度场 ---
    axes[0, 0].set_title(f"Ground Truth - U Velocity (Relative Err: {error_u_relative*100:.2f}%)")
    fig.colorbar(axes[0, 0].pcolormesh(grid_x_raw, grid_y_raw, gt_phys[ch_u], shading='gouraud', cmap=cmap), ax=axes[0, 0])
    
    axes[0, 1].set_title(f"Prediction - U Velocity (Channel $L_2$: {l2_error_u*100:.2f}%)")
    fig.colorbar(axes[0, 1].pcolormesh(grid_x_raw, grid_y_raw, pred_phys[ch_u], shading='gouraud', cmap=cmap), ax=axes[0, 1])
    
    # --- 行 2：P 压力场 ---
    axes[1, 0].set_title(f"Ground Truth - Pressure (Relative Err: {error_p_relative*100:.2f}%)")
    fig.colorbar(axes[1, 0].pcolormesh(grid_x_raw, grid_y_raw, gt_phys[ch_p], shading='gouraud', cmap=cmap), ax=axes[1, 0])
    
    axes[1, 1].set_title(f"Prediction - Pressure (Channel $L_2$: {l2_error_p*100:.2f}%)")
    fig.colorbar(axes[1, 1].pcolormesh(grid_x_raw, grid_y_raw, pred_phys[ch_p], shading='gouraud', cmap=cmap), ax=axes[1, 1])
    
    # --- 行 3：绝对误差场 + 红十字十字星圈出最大误差最高点 ---
    # U 速度绝对误差子图
    axes[2, 0].set_title("Absolute Error - U Velocity")
    im_err_u = axes[2, 0].pcolormesh(grid_x_raw, grid_y_raw, err_phys[ch_u], shading='gouraud', cmap=err_cmap)
    fig.colorbar(im_err_u, ax=axes[2, 0])
    axes[2, 0].scatter(max_err_x_u, max_err_y_u, color='red', marker='x', s=120, linewidths=2.5, label='Max Error Point', zorder=5)
    circle_u = plt.Circle((max_err_x_u, max_err_y_u), 0.08, color='red', fill=False, linewidth=1.5, linestyle='--', zorder=5)
    axes[2, 0].add_patch(circle_u)
    axes[2, 0].legend(loc='upper right', fontsize=8)
    
    # 压力绝对误差子图
    axes[2, 1].set_title("Absolute Error - Pressure")
    im_err_p = axes[2, 1].pcolormesh(grid_x_raw, grid_y_raw, err_phys[ch_p], shading='gouraud', cmap=err_cmap)
    fig.colorbar(im_err_p, ax=axes[2, 1])
    axes[2, 1].scatter(max_err_x_p, max_err_y_p, color='red', marker='x', s=120, linewidths=2.5, label='Max Error Point', zorder=5)
    circle_p = plt.Circle((max_err_x_p, max_err_y_p), 0.08, color='red', fill=False, linewidth=1.5, linestyle='--', zorder=5)
    axes[2, 1].add_patch(circle_p)
    axes[2, 1].legend(loc='upper right', fontsize=8)
    
    for ax in axes.flatten():
        ax.set_aspect('equal', adjustable='box')
        
    plt.tight_layout()
    plt.show()