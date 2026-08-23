#usr/bin/python3
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 🌟 换成你高保真自适应长方形去噪网络
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
    
    # 🎯 动态自适应大网格尺寸，直接在原生高分辨率空间里注入高斯噪声开始演进
    h, w = grid_x.shape[1], grid_x.shape[2]
    x_t = torch.randn((1, 4, h, w), device=device)
    
    # 为了防止控制台日志被刷屏，批量遍历时关闭内部的 tqdm 进度条
    for t_idx in reversed(range(timesteps)):
        t = torch.full((1,), t_idx, device=device, dtype=torch.long)
        x0_pred = model(x_t, grid_x, grid_y, t, target_cond)
        ab_t = alpha_bar[t_idx]
        
        if t_idx > 0:
            ab_t_prev = alpha_bar[t_idx - 1]
            weight_x0 = torch.sqrt(ab_t_prev) * beta[t_idx] / (1.0 - ab_t)
            weight_xt = torch.sqrt(alpha[t_idx]) * (1.0 - ab_t_prev) / (1.0 - ab_t)
            
            mean = weight_x0 * x0_pred + weight_xt * x_t
            var = (1.0 - ab_t_prev) / (1.0 - ab_t) * beta[t_idx]
            noise = torch.randn_like(x_t)
            
            x_t = mean + torch.sqrt(var) * noise
        else:
            x_t = x0_pred
            
    return x_t

def relative_error_calc(pred, true, eps=1e-8):
    """等价于原有 CFDLib 的单通道相对误差函数"""
    return np.sum(np.abs(pred - true)) / (np.sum(np.abs(true)) + eps)

# ==========================================
# 2. 自动化批量检索与评估主程序
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动 (60x301) 高保真变尺寸全量测试集清算引擎: {device}")
    
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else "."
    # 🎯 指向高保真专用的权重文件与参考数据集路径
    weights_path = os.path.abspath(os.path.join(current_dir, "../Results/airfoil_diffusion_highres_ablation_nomask_ep5000.pth"))
    norm_path = os.path.abspath(os.path.join(current_dir, "../Results/normalization_factors_highres.npz"))
    
    train_data_path = os.path.abspath(os.path.join(current_dir, "../Results/HighRes_airfoil_train.npz"))
    test_data_path = os.path.abspath(os.path.join(current_dir, "../Results/HighRes_airfoil_test.npz"))
    
    # 🎯 引入自适应高精度架构网络
    model = HighResDiffusionUNet().to(device)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("✅ (60x301) 高保真稳态物理模型权重载入完毕。")
    else:
        raise FileNotFoundError(f"❌ 找不到模型权重文件: {weights_path}")
    
    train_data = np.load(train_data_path)
    test_data = np.load(test_data_path)
    norm_factors = np.load(norm_path)
    
    f_min, f_max = norm_factors['fields_min'], norm_factors['fields_max']
    l_min, l_max = norm_factors['label_min'], norm_factors['label_max']
    grid_x_raw, grid_y_raw = train_data['grid_x'], train_data['grid_y']
    
    # 提取测试集的总样本数
    num_test_samples = test_data['x'].shape[0]
    print(f"📊 扫描到高清测试集总计工况数量: {num_test_samples} 个。开始全自动化批处理推理...\n")
    
    # 初始化三个通道的单通道误差累加器，以及 Table 4 综合空间误差全集列表
    error_u_sum, error_v_sum, error_p_sum = 0.0, 0.0, 0.0
    case_combined_errors = []
    
    # 用于记录最优秀和最差工况的流场，方便最后单独画图
    best_case_info = {"error": float('inf'), "gt": None, "pred": None, "phys": None}
    worst_case_info = {"error": float('-inf'), "gt": None, "pred": None, "phys": None}
    
    # 建立反归一化闭包函数
    def denormalize(arr):
        phys_0_1 = (arr + 1.0) / 2.0
        f_min_c = f_min[0, :, 0, 0].reshape(4, 1, 1)
        f_max_c = f_max[0, :, 0, 0].reshape(4, 1, 1)
        return phys_0_1 * (f_max_c - f_min_c) + f_min_c

    # 严格对齐 [U, V, Rho, P] 的 4 通道物理排布
    ch_u, ch_v, ch_p = 0, 1, 3
    eps = 1e-5
    
    for i in tqdm(range(num_test_samples), desc="🌪️ 高清测试集全量样本正在通过 Diffusion 生成演进"):
        # 1. 提取当前测试样本的归一化控制标签与真实流场
        target_norm = test_data['y'][i]
        gt_fields_norm = test_data['x'][i]
        
        # 2. 物理参数反推，用于打印记录
        cond_phys = target_norm * (l_max[0] - l_min[0] + 1e-8) + l_min[0]
        
        # 3. 转换为 PyTorch 张量形态准备喂入网络
        cond_tensor = torch.tensor(target_norm, dtype=torch.float32, device=device).unsqueeze(0)
        grid_x_tensor = torch.tensor(grid_x_raw, dtype=torch.float32, device=device).unsqueeze(0)
        grid_y_tensor = torch.tensor(grid_y_raw, dtype=torch.float32, device=device).unsqueeze(0)
        
        # 4. 执行高保真无插值反向去噪采样
        pred_field_norm = sample_flow_field_scheme_b(model, device, grid_x_tensor, grid_y_tensor, cond_tensor)
        
        # 5. 双重物理领域反归一化
        gt_phys = denormalize(gt_fields_norm)
        pred_phys = pred_field_norm.cpu().numpy()[0]
        pred_phys = ((pred_phys + 1.0) / 2.0) * (f_max[0,:,0,0].reshape(4,1,1) - f_min[0,:,0,0].reshape(4,1,1)) + f_min[0,:,0,0].reshape(4,1,1)
        
        # 6. 计算单通道传统的离散相对误差
        error_u = relative_error_calc(pred_phys[ch_u].flatten(), gt_phys[ch_u].flatten(), eps)
        error_v = relative_error_calc(pred_phys[ch_v].flatten(), gt_phys[ch_v].flatten(), eps)
        error_p = relative_error_calc(pred_phys[ch_p].flatten(), gt_phys[ch_p].flatten(), eps)
        
        error_u_sum += error_u
        error_v_sum += error_v
        error_p_sum += error_p
        
        # 7. 计算学术界 Table 4 所需的 3 通道空间融合相对 L2 误差 (%)
        u_pred_f, u_true_f = pred_phys[ch_u].flatten(), gt_phys[ch_u].flatten()
        v_pred_f, v_true_f = pred_phys[ch_v].flatten(), gt_phys[ch_v].flatten()
        p_pred_f, p_true_f = pred_phys[ch_p].flatten(), gt_phys[ch_p].flatten()
        
        res_sq = np.sum((u_pred_f - u_true_f)**2) + np.sum((v_pred_f - v_true_f)**2) + np.sum((p_pred_f - p_true_f)**2)
        den_sq = np.sum(u_true_f**2) + np.sum(v_true_f**2) + np.sum(p_true_f**2)
        case_l2_error = np.sqrt(res_sq / den_sq) * 100
        case_combined_errors.append(case_l2_error)
        
        # 8. 实时追踪记录最大与最小误差工况的详细状态
        if case_l2_error < best_case_info["error"]:
            best_case_info.update({"error": case_l2_error, "gt": gt_phys, "pred": pred_phys, "phys": cond_phys})
        if case_l2_error > worst_case_info["error"]:
            worst_case_info.update({"error": case_l2_error, "gt": gt_phys, "pred": pred_phys, "phys": cond_phys})

    # =================================================================
    # 3. 📊 核心汇总：最终的 Table 4 指标宏观统计与完美打印
    # =================================================================
    case_combined_errors = np.array(case_combined_errors)
    
    mean_error_t4 = np.mean(case_combined_errors)
    min_error_t4 = np.min(case_combined_errors)
    max_error_t4 = np.max(case_combined_errors)
    
    print('\n' + '='*65)
    print(' 【Table 4 对应指标】Physics-Diffusion 高清稳态流场综合误差报告 (Error %):')
    print('='*65)
    print(f' >> 平均误差 (Mean Error) : {mean_error_t4:.4f}%')
    print(f' >> 最小误差 (Min Error)  : {min_error_t4:.4f}%  (工况: Mach={best_case_info["phys"][0]:.3f}, AoA={best_case_info["phys"][1]:.1f}°)')
    print(f' >> 最大误差 (Max Error)  : {max_error_t4:.4f}%  (工况: Mach={worst_case_info["phys"][0]:.3f}, AoA={worst_case_info["phys"][1]:.1f}°)')
    print('='*65 + '\n')
    
    # 单通道传统离散平均误差备份打印
    print('备份单通道离散相对误差平均值 (与高精度 U-Net/GPR 基准对齐):')
    print(f'Mean Error u: {error_u_sum / num_test_samples:.6e}')
    print(f'Mean Error v: {error_v_sum / num_test_samples:.6e}')
    print(f'Mean Error p: {error_p_sum / num_test_samples:.6e}\n')

    # ==========================================
    # 4. 🎨 终极可视化：自动绘制最优秀工况的 3x2 矩阵图
    # ==========================================
    print(f"🎨 正在渲染高清测试集表现最佳【Min Error Case ({min_error_t4:.2f}%)】的物理流场...")
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f"High-Res BEST Prediction Case (Table 4 L2: {min_error_t4:.3f}%)\nMach={best_case_info['phys'][0]:.3f}, AoA={best_case_info['phys'][1]:.1f}°", fontsize=13, fontweight='bold')
    
    levels = 50
    cmap, err_cmap = 'jet', 'magma'
    bp_gt, bp_pred = best_case_info["gt"], best_case_info["pred"]
    bp_err = np.abs(bp_gt - bp_pred)
    
    # U 速度场
    axes[0, 0].set_title("Ground Truth - U Velocity")
    fig.colorbar(axes[0, 0].contourf(grid_x_raw, grid_y_raw, bp_gt[ch_u], levels=levels, cmap=cmap), ax=axes[0, 0])
    axes[0, 1].set_title("Prediction - U Velocity")
    fig.colorbar(axes[0, 1].contourf(grid_x_raw, grid_y_raw, bp_pred[ch_u], levels=levels, cmap=cmap), ax=axes[0, 1])
    
    # P 压力场
    axes[1, 0].set_title("Ground Truth - Pressure")
    fig.colorbar(axes[1, 0].contourf(grid_x_raw, grid_y_raw, bp_gt[ch_p], levels=levels, cmap=cmap), ax=axes[1, 0])
    axes[1, 1].set_title("Prediction - Pressure")
    fig.colorbar(axes[1, 1].contourf(grid_x_raw, grid_y_raw, bp_pred[ch_p], levels=levels, cmap=cmap), ax=axes[1, 1])
    
    # 绝对误差
    axes[2, 0].set_title("Absolute Error - U Velocity")
    fig.colorbar(axes[2, 0].contourf(grid_x_raw, grid_y_raw, bp_err[ch_u], levels=levels, cmap=err_cmap), ax=axes[2, 0])
    axes[2, 1].set_title("Absolute Error - Pressure")
    fig.colorbar(axes[2, 1].contourf(grid_x_raw, grid_y_raw, bp_err[ch_p], levels=levels, cmap=err_cmap), ax=axes[2, 1])
    
    for ax in axes.flatten(): ax.axis('equal')
    plt.tight_layout()
    plt.show()