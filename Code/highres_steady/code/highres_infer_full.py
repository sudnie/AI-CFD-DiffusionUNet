#!/usr/bin/python3
import os
import torch
import numpy as np
import pandas as pd          # 新增：用于保存详细指标
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

# ==========================================
# 1. 原相对误差函数（单通道，绝对值归一化）
# ==========================================
def relative_error_calc(pred, true, eps=1e-8):
    return np.sum(np.abs(pred - true)) / (np.sum(np.abs(true)) + eps)

# ==========================================
# 2. 新增完整指标计算函数（复刻文件2）
# ==========================================
def calculate_full_metrics(gt_u, gt_v, gt_p, pred_u, pred_v, pred_p, eps=1e-8):
    """
    输入均为 flatten 一维 numpy 数组
    返回字典：包含 Combined L2 (%), 各分量 L2 (%), RMSE, MAE, NRMSE, MSE
    """
    # ---- 1. 标准 L2 相对误差（百分比） ----
    l2_u = np.linalg.norm(pred_u - gt_u) / (np.linalg.norm(gt_u) + eps) * 100
    l2_v = np.linalg.norm(pred_v - gt_v) / (np.linalg.norm(gt_v) + eps) * 100
    l2_p = np.linalg.norm(pred_p - gt_p) / (np.linalg.norm(gt_p) + eps) * 100

    # ---- 2. 组合相对误差 (Combined L2, 百分比) ----
    num = np.sum((pred_u - gt_u)**2) + np.sum((pred_v - gt_v)**2) + np.sum((pred_p - gt_p)**2)
    den = np.sum(gt_u**2) + np.sum(gt_v**2) + np.sum(gt_p**2)
    comb_l2 = np.sqrt(num / (den + eps)) * 100

    # ---- 3. RMSE (量纲) ----
    rmse_u = np.sqrt(np.mean((pred_u - gt_u)**2))
    rmse_v = np.sqrt(np.mean((pred_v - gt_v)**2))
    rmse_p = np.sqrt(np.mean((pred_p - gt_p)**2))

    # ---- 4. MAE (量纲) ----
    mae_u = np.mean(np.abs(pred_u - gt_u))
    mae_v = np.mean(np.abs(pred_v - gt_v))
    mae_p = np.mean(np.abs(pred_p - gt_p))

    # ---- 5. NRMSE (等价于 CFDLib.relative_error, 减均值归一化) ----
    def nrmse(gt, pred):
        numerator = np.mean((pred - gt) ** 2)
        denominator = np.mean((gt - np.mean(gt)) ** 2)
        return np.sqrt(numerator / (denominator + eps))

    nrmse_u = nrmse(gt_u, pred_u)
    nrmse_v = nrmse(gt_v, pred_v)
    nrmse_p = nrmse(gt_p, pred_p)

    # ---- 6. MSE (等价于 CFDLib.mean_squared_error) ----
    mse_u = np.mean((pred_u - gt_u) ** 2)
    mse_v = np.mean((pred_v - gt_v) ** 2)
    mse_p = np.mean((pred_p - gt_p) ** 2)

    return {
        "comb_l2": comb_l2,
        "l2_u": l2_u, "l2_v": l2_v, "l2_p": l2_p,
        "rmse_u": rmse_u, "rmse_v": rmse_v, "rmse_p": rmse_p,
        "mae_u": mae_u, "mae_v": mae_v, "mae_p": mae_p,
        "nrmse_u": nrmse_u, "nrmse_v": nrmse_v, "nrmse_p": nrmse_p,
        "mse_u": mse_u, "mse_v": mse_v, "mse_p": mse_p,
    }

# ==========================================
# 3. 自动化批量检索与评估主程序
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动 (60x301) 高保真变尺寸全量测试集清算引擎: {device}")
    
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else "."
    # 🎯 指向高保真专用的权重文件与参考数据集路径
    weights_path = os.path.abspath(os.path.join(current_dir, "../Results/airfoil_diffusion_mse_only_ep2000.pth"))
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
    
    # ============ 初始化各误差累加器 ============
    error_u_sum, error_v_sum, error_p_sum = 0.0, 0.0, 0.0      # 原有单通道相对误差累加
    all_metrics = []                                            # 存储每一帧的完整指标字典
    
    # 用于记录最优秀和最差工况的流场（基于Table4综合L2）
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
        
        # 提取各通道flatten数据
        u_true = gt_phys[ch_u].flatten()
        v_true = gt_phys[ch_v].flatten()
        p_true = gt_phys[ch_p].flatten()
        u_pred = pred_phys[ch_u].flatten()
        v_pred = pred_phys[ch_v].flatten()
        p_pred = pred_phys[ch_p].flatten()
        
        # ----- 6. 原有单通道相对误差（绝对值归一化）-----
        error_u = relative_error_calc(u_pred, u_true, eps)
        error_v = relative_error_calc(v_pred, v_true, eps)
        error_p = relative_error_calc(p_pred, p_true, eps)
        error_u_sum += error_u
        error_v_sum += error_v
        error_p_sum += error_p
        
        # ----- 7. Table 4 综合 L2 误差 (%) -----
        res_sq = np.sum((u_pred - u_true)**2) + np.sum((v_pred - v_true)**2) + np.sum((p_pred - p_true)**2)
        den_sq = np.sum(u_true**2) + np.sum(v_true**2) + np.sum(p_true**2)
        case_l2_error = np.sqrt(res_sq / (den_sq + eps)) * 100
        # 记录用于追踪最佳/最差工况（基于此指标）
        if case_l2_error < best_case_info["error"]:
            best_case_info.update({"error": case_l2_error, "gt": gt_phys, "pred": pred_phys, "phys": cond_phys})
        if case_l2_error > worst_case_info["error"]:
            worst_case_info.update({"error": case_l2_error, "gt": gt_phys, "pred": pred_phys, "phys": cond_phys})
        
        # ----- 8. 新增完整指标计算（文件2风格）-----
        metrics = calculate_full_metrics(
            gt_u=u_true, gt_v=v_true, gt_p=p_true,
            pred_u=u_pred, pred_v=v_pred, pred_p=p_pred,
            eps=eps
        )
        # 添加工况信息
        metrics.update({
            "sample_idx": i,
            "mach": float(cond_phys[0]),
            "aoa": float(cond_phys[1]),
            "table4_l2": case_l2_error,   # 同时保存Table4综合L2，方便后续对比
        })
        all_metrics.append(metrics)

    # =================================================================
    # 3. 📊 核心汇总：最终的统计报表（包括原Table4和新增全套指标）
    # =================================================================
    # 转换为DataFrame便于统计
    df = pd.DataFrame(all_metrics)
    
    # 保存CSV（可选）
    csv_path = os.path.join(current_dir, "../Results/highres_full_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n📁 详细评估指标已保存至: {csv_path}")

    # ---------- 打印原有Table4综合L2统计 ----------
    print('\n' + '='*65)
    print(' 【Table 4 对应指标】Physics-Diffusion 高清稳态流场综合误差报告 (Error %):')
    print('='*65)
    print(f' >> 平均误差 (Mean Error) : {df["table4_l2"].mean():.4f}% ± {df["table4_l2"].std():.4f}%')
    print(f' >> 最小误差 (Min Error)  : {df["table4_l2"].min():.4f}%  (工况: Mach={best_case_info["phys"][0]:.3f}, AoA={best_case_info["phys"][1]:.1f}°)')
    print(f' >> 最大误差 (Max Error)  : {df["table4_l2"].max():.4f}%  (工况: Mach={worst_case_info["phys"][0]:.3f}, AoA={worst_case_info["phys"][1]:.1f}°)')
    print('='*65 + '\n')
    
    # ---------- 原有单通道相对误差（绝对值归一化） ----------
    print('备份单通道离散相对误差平均值 (与高精度 U-Net/GPR 基准对齐):')
    print(f'Mean Error u: {error_u_sum / num_test_samples:.6e}')
    print(f'Mean Error v: {error_v_sum / num_test_samples:.6e}')
    print(f'Mean Error p: {error_p_sum / num_test_samples:.6e}\n')
    
    # ---------- 新增全套指标统计报表（完全对标文件2） ----------
    print("="*74)
    print("📊 [完整物理场评估报告] (HighRes Diffusion 测试集)")
    print("="*74)
    print(f"  - 综合 L2 (Combined L2) : {df['comb_l2'].mean():6.3f}% ± {df['comb_l2'].std():6.3f}%")
    print("  ------------------------------------------------------------------------")
    print(f"  - U 速度场 : L2 = {df['l2_u'].mean():6.3f}% | RMSE = {df['rmse_u'].mean():.4f} m/s | MAE = {df['mae_u'].mean():.4f} m/s")
    print(f"  - V 速度场 : L2 = {df['l2_v'].mean():6.3f}% | RMSE = {df['rmse_v'].mean():.4f} m/s | MAE = {df['mae_v'].mean():.4f} m/s")
    print(f"  - P 压力场 : L2 = {df['l2_p'].mean():6.3f}% | RMSE = {df['rmse_p'].mean():.4f} Pa  | MAE = {df['mae_p'].mean():.4f} Pa")
    print("="*74)
    print(f"  - NRMSE (方差归一化) : U = {df['nrmse_u'].mean():.6f}, V = {df['nrmse_v'].mean():.6f}, P = {df['nrmse_p'].mean():.6f}")
    print(f"  - MSE  (量纲平方)  : U = {df['mse_u'].mean():.6f}, V = {df['mse_v'].mean():.6f}, P = {df['mse_p'].mean():.6f}")
    print("="*74)

    # 可选：找出最佳/最差工况基于综合L2
    best_idx = df['comb_l2'].idxmin()
    worst_idx = df['comb_l2'].idxmax()
    print(f"\n🏆 最佳综合L2样本: Mach={df.loc[best_idx, 'mach']:.3f}, AoA={df.loc[best_idx, 'aoa']:.1f}°, L2={df.loc[best_idx, 'comb_l2']:.3f}%")
    print(f"⚠️  最差综合L2样本: Mach={df.loc[worst_idx, 'mach']:.3f}, AoA={df.loc[worst_idx, 'aoa']:.1f}°, L2={df.loc[worst_idx, 'comb_l2']:.3f}%")
    print("="*74)

    # ==========================================
    # 4. 🎨 终极可视化：自动绘制最优秀工况的 3x2 矩阵图（基于Table4最佳）
    # ==========================================
    print(f"\n🎨 正在渲染高清测试集表现最佳【Min Error Case ({best_case_info['error']:.2f}%)】的物理流场...")
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f"High-Res BEST Prediction Case (Table 4 L2: {best_case_info['error']:.3f}%)\nMach={best_case_info['phys'][0]:.3f}, AoA={best_case_info['phys'][1]:.1f}°", fontsize=13, fontweight='bold')
    
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
    plt.savefig(os.path.join(current_dir, "../Results/highres_best_case_vis.png"), dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ 可视化已保存为 highres_best_case_vis.png")