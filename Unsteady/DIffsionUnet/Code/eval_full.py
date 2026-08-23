#!/usr/bin/python3
"""
전체 크기 (145x689) C-grid DiffusionUNet 초고속 추론 및 물리 평가 스크립트
기능: DDIM 50단계 빠른 샘플링 + 물리 역정규화 + 다차원 오차 통계 (L2, RMSE, MAE) + 고밀도 등치선 시각화
"""

import os
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from model_utils import DiffusionUNet


# ==========================================
# 1. DDIM 결정론적 빠른 샘플러 (DDIM Sampler)
# ==========================================
@torch.no_grad()
def sample_ddim_batch(model, device, grid_x, grid_y, target_cond_batch, total_timesteps=1000, ddim_steps=50):
    model.eval()
    bsz = target_cond_batch.shape[0]
    H, W = grid_x.shape[-2], grid_x.shape[-1]
    
    beta = torch.linspace(1e-4, 0.02, total_timesteps).to(device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    
    times = torch.linspace(total_timesteps - 1, 0, ddim_steps, dtype=torch.long, device=device)
    x_t = torch.randn((bsz, 4, H, W), device=device)
    
    grid_x_b = grid_x.repeat(bsz, 1, 1, 1) if grid_x.shape[0] == 1 else grid_x
    grid_y_b = grid_y.repeat(bsz, 1, 1, 1) if grid_y.shape[0] == 1 else grid_y

    for i in range(len(times)):
        t_idx = times[i]
        t = torch.full((bsz,), t_idx, device=device, dtype=torch.long)
        
        # 模型前向推理得到去噪后的预测场 x0
        x0_pred = model(x_t, grid_x_b, grid_y_b, t, target_cond_batch)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        
        if i == len(times) - 1:
            x_t = x0_pred
            break
            
        t_next_idx = times[i + 1]
        ab_t = alpha_bar[t_idx]
        ab_next = alpha_bar[t_next_idx]
        
        eps_pred = (x_t - torch.sqrt(ab_t) * x0_pred) / (torch.sqrt(1.0 - ab_t) + 1e-8)
        x_t = torch.sqrt(ab_next) * x0_pred + torch.sqrt(1.0 - ab_next) * eps_pred
            
    return x_t


# ==========================================
# 2. 물리량 역정규화 및 다차원 오차 계산
# ==========================================
def denormalize_batch(field_tensor, f_min, f_max):
    if isinstance(field_tensor, torch.Tensor):
        arr = field_tensor.detach().cpu().numpy()
    else:
        arr = np.array(field_tensor)
        
    phys_0_1 = (arr + 1.0) / 2.0
    f_min_c = f_min.reshape(1, 4, 1, 1)
    f_max_c = f_max.reshape(1, 4, 1, 1)
    return phys_0_1 * (f_max_c - f_min_c) + f_min_c


def calculate_metrics_single(gt_single, pred_single):
    eps = 1e-5
    ch_rho, ch_u, ch_v, ch_p = 0, 1, 2, 3
    
    # 1. 상대 L2 오차 (%)
    l2_u = np.linalg.norm(gt_single[ch_u] - pred_single[ch_u]) / (np.linalg.norm(gt_single[ch_u]) + eps) * 100
    l2_v = np.linalg.norm(gt_single[ch_v] - pred_single[ch_v]) / (np.linalg.norm(gt_single[ch_v]) + eps) * 100
    l2_p = np.linalg.norm(gt_single[ch_p] - pred_single[ch_p]) / (np.linalg.norm(gt_single[ch_p]) + eps) * 100
    
    u_p, u_t = pred_single[ch_u].flatten(), gt_single[ch_u].flatten()
    v_p, v_t = pred_single[ch_v].flatten(), gt_single[ch_v].flatten()
    p_p, p_t = pred_single[ch_p].flatten(), gt_single[ch_p].flatten()
    
    res_sq = np.sum((u_p - u_t)**2) + np.sum((v_p - v_t)**2) + np.sum((p_p - p_t)**2)
    den_sq = np.sum(u_t**2) + np.sum(v_t**2) + np.sum(p_t**2)
    comb_l2 = np.sqrt(res_sq / (den_sq + eps)) * 100

    # 2. 평균 제곱근 오차 (RMSE)
    rmse_u = np.sqrt(np.mean((gt_single[ch_u] - pred_single[ch_u])**2))
    rmse_v = np.sqrt(np.mean((gt_single[ch_v] - pred_single[ch_v])**2))
    rmse_p = np.sqrt(np.mean((gt_single[ch_p] - pred_single[ch_p])**2))

    # 3. 평균 절대 오차 (MAE)
    mae_u = np.mean(np.abs(gt_single[ch_u] - pred_single[ch_u]))
    mae_v = np.mean(np.abs(gt_single[ch_v] - pred_single[ch_v]))
    mae_p = np.mean(np.abs(gt_single[ch_p] - pred_single[ch_p]))

    return {
        "comb_l2": comb_l2, "l2_u": l2_u, "l2_v": l2_v, "l2_p": l2_p,
        "rmse_u": rmse_u, "rmse_v": rmse_v, "rmse_p": rmse_p,
        "mae_u": mae_u, "mae_v": mae_v, "mae_p": mae_p
    }


# ==========================================
# 3. 데이터셋 평가 및 추론 루프
# ==========================================
def evaluate_dataset(dataset_name, data_npz, model, device, grid_x_tensor, grid_y_tensor, f_min, f_max, l_min, l_max, ddim_steps=50, batch_size=4):
    print(f"\n🚀 데이터셋 추론 평가 시작: 【{dataset_name}】 (총 샘플 수: {len(data_npz['x'])})")
    
    x_data = data_npz['x']
    y_data = data_npz['y']
    num_samples = len(x_data)
    results = []
    
    all_gt_phys = []
    all_pred_phys = []
    
    start_time = time.time()
    for i in tqdm(range(0, num_samples, batch_size), desc=f"Inferencing {dataset_name}"):
        batch_x_norm = x_data[i:i+batch_size]
        batch_y_norm = y_data[i:i+batch_size]
        
        cond_tensor = torch.tensor(batch_y_norm, dtype=torch.float32, device=device)
        if cond_tensor.dim() == 1:
            cond_tensor = cond_tensor.unsqueeze(1)
            
        pred_norm_ddim = sample_ddim_batch(
            model, device, grid_x_tensor, grid_y_tensor, cond_tensor, 
            total_timesteps=1000, ddim_steps=ddim_steps
        )
        
        gt_phys = denormalize_batch(batch_x_norm, f_min, f_max)
        pred_phys = denormalize_batch(pred_norm_ddim, f_min, f_max)
        
        all_gt_phys.append(gt_phys)
        all_pred_phys.append(pred_phys)
        
        for j in range(gt_phys.shape[0]):
            idx = i + j
            # 反归一化条件标签以获取真实物理时间
            time_val = (batch_y_norm[j, 2] * (l_max[0, 2] - l_min[0, 2] + 1e-8) + l_min[0, 2]).item() if batch_y_norm.ndim > 1 else (batch_y_norm[j] * (l_max - l_min + 1e-8) + l_min).item()
            metrics = calculate_metrics_single(gt_phys[j], pred_phys[j])
            
            entry = {
                "sample_idx": idx,
                "time_step": time_val,
                "comb_l2 (%)": metrics["comb_l2"],
                "l2_u (%)": metrics["l2_u"], "l2_v (%)": metrics["l2_v"], "l2_p (%)": metrics["l2_p"],
                "rmse_u": metrics["rmse_u"], "rmse_v": metrics["rmse_v"], "rmse_p": metrics["rmse_p"],
                "mae_u": metrics["mae_u"], "mae_v": metrics["mae_v"], "mae_p": metrics["mae_p"]
            }
            results.append(entry)
            
    total_time = time.time() - start_time
    df = pd.DataFrame(results)
    
    all_gt_phys = np.concatenate(all_gt_phys, axis=0)
    all_pred_phys = np.concatenate(all_pred_phys, axis=0)
    
    print(f"✅ {dataset_name} 추론 완료! 소요 시간: {total_time:.2f}초 (샘플당 평균: {total_time/num_samples:.3f}초)")
    return df, all_gt_phys, all_pred_phys


# ==========================================
# 4. 高精流场云图对比绘制函数 (高密度等치선 叠加)
# ==========================================
def plot_contour_comparison(
    grid_x_raw, grid_y_raw, gt_sample, pred_sample, time_val, save_path
):
    ch_u, ch_v, ch_p = 1, 2, 3

    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True, sharey=True)
    fig.suptitle(
        f"Full Grid (145x689) High-Density Contour Lines Benchmark (Time = {time_val:.1f})",
        fontsize=14,
        fontweight="bold",
    )

    cmap_field = "jet"
    cmap_err = "magma"

    channels = [
        ("U Velocity", ch_u, "m/s"),
        ("V Velocity", ch_v, "m/s"),
        ("Pressure", ch_p, "Pa"),
    ]

    for row, (name, ch_idx, unit) in enumerate(channels):
        gt = gt_sample[ch_idx]
        pred = pred_sample[ch_idx]
        abs_err = np.abs(gt - pred)

        val_min = min(gt.min(), pred.min())
        val_max = max(gt.max(), pred.max())
        levels_dense = np.linspace(val_min, val_max, 35)

        # ----------------------------------------------------
        # 1. Ground Truth
        # ----------------------------------------------------
        ax_gt = axes[row, 0]
        c0 = ax_gt.pcolormesh(
            grid_x_raw, grid_y_raw, gt, shading="gouraud", cmap=cmap_field
        )
        ax_gt.contour(
            grid_x_raw,
            grid_y_raw,
            gt,
            levels=levels_dense,
            colors="black",
            linewidths=0.35,
            alpha=0.55,
        )
        ax_gt.contour(
            grid_x_raw,
            grid_y_raw,
            gt,
            levels=levels_dense[::5],
            colors="black",
            linewidths=0.7,
            alpha=0.85,
        )
        ax_gt.set_title(f"Ground Truth - {name}")
        fig.colorbar(c0, ax=ax_gt, label=unit)

        # ----------------------------------------------------
        # 2. DDIM Prediction
        # ----------------------------------------------------
        ax_pred = axes[row, 1]
        c1 = ax_pred.pcolormesh(
            grid_x_raw, grid_y_raw, pred, shading="gouraud", cmap=cmap_field
        )
        ax_pred.contour(
            grid_x_raw,
            grid_y_raw,
            pred,
            levels=levels_dense,
            colors="black",
            linewidths=0.35,
            alpha=0.55,
        )
        ax_pred.contour(
            grid_x_raw,
            grid_y_raw,
            pred,
            levels=levels_dense[::5],
            colors="black",
            linewidths=0.7,
            alpha=0.85,
        )
        ax_pred.set_title(f"DDIM Prediction - {name}")
        fig.colorbar(c1, ax=ax_pred, label=unit)

        # ----------------------------------------------------
        # 3. Absolute Error Map
        # ----------------------------------------------------
        ax_err = axes[row, 2]
        c2 = ax_err.pcolormesh(
            grid_x_raw, grid_y_raw, abs_err, shading="gouraud", cmap=cmap_err
        )
        err_levels = np.linspace(0, abs_err.max() + 1e-8, 20)
        ax_err.contour(
            grid_x_raw,
            grid_y_raw,
            abs_err,
            levels=err_levels,
            colors="white",
            linewidths=0.3,
            alpha=0.6,
        )

        mean_err = np.mean(abs_err)
        max_err = np.max(abs_err)
        ax_err.set_title(
            f"Abs Error - {name} (MAE: {mean_err:.4f}, Max: {max_err:.4f})"
        )
        fig.colorbar(c2, ax=ax_err, label=unit)

    for ax in axes.flatten():
        ax.plot(
            grid_x_raw[0, :],
            grid_y_raw[0, :],
            "k-",
            linewidth=1.2,
            zorder=10,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"🖼️ 고밀도 등치선이 포함된 정밀 등고선도가 저장되었습니다: {save_path}")
    plt.show()


# ==========================================
# 5. 메인 실행 진입점
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🌟 전체 크기 (145x689) Diffusion 추론 및 평가 프로세스 시작 | 장치: {device}")

    DDIM_STEPS = 50
    BATCH_SIZE = 4

    current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
    results_dir = os.path.abspath(os.path.join(current_dir, "../Results"))

    # 权重路径匹配（支持续训保存的各种命名）
    weights_path = os.path.join(results_dir, "airfoil_diffusion_cgrid_final.pth")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(results_dir, "airfoil_diffusion_cgrid_ep3000.pth")

    norm_path = os.path.join(results_dir, "normalization_factors_full.npz")
    train_data_path = os.path.join(results_dir, "Diffusion_airfoil_unsteady_full_train.npz")
    test_data_path = os.path.join(results_dir, "Diffusion_airfoil_unsteady_full_test.npz")

    model = DiffusionUNet(flow_ch=4, coord_ch=2, cond_dim=128, base_ch=48).to(device)

    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        # 兼容完整 Checkpoint 字典结构与纯 state_dict 结构
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"✅ 权重模型 로드 성공: {weights_path}")
    else:
        raise FileNotFoundError(f"❌ 가중치 파일을 찾을 수 없습니다: {weights_path}")

    train_data = np.load(train_data_path)
    test_data = np.load(test_data_path)
    norm_factors = np.load(norm_path)

    f_min, f_max = norm_factors["fields_min"], norm_factors["fields_max"]
    l_min, l_max = norm_factors["label_min"], norm_factors["label_max"]

    grid_x_raw, grid_y_raw = train_data["grid_x"], train_data["grid_y"]
    grid_x_tensor = torch.tensor(grid_x_raw, dtype=torch.float32, device=device)
    grid_y_tensor = torch.tensor(grid_y_raw, dtype=torch.float32, device=device)

    grid_x_tensor = 2.0 * (grid_x_tensor - grid_x_tensor.min()) / (grid_x_tensor.max() - grid_x_tensor.min() + 1e-8) - 1.0
    grid_y_tensor = 2.0 * (grid_y_tensor - grid_y_tensor.min()) / (grid_y_tensor.max() - grid_y_tensor.min() + 1e-8) - 1.0

    if grid_x_tensor.dim() == 2:
        grid_x_tensor = grid_x_tensor.unsqueeze(0).unsqueeze(0)
        grid_y_tensor = grid_y_tensor.unsqueeze(0).unsqueeze(0)

    # 运行数据集批量推理评估
    df_train, train_gt, train_pred = evaluate_dataset("Train Dataset", train_data, model, device, grid_x_tensor, grid_y_tensor, f_min, f_max, l_min, l_max, ddim_steps=DDIM_STEPS, batch_size=BATCH_SIZE)
    df_test, test_gt, test_pred = evaluate_dataset("Test Dataset", test_data, model, device, grid_x_tensor, grid_y_tensor, f_min, f_max, l_min, l_max, ddim_steps=DDIM_STEPS, batch_size=BATCH_SIZE)

    # 保存评估结果 CSV
    df_train.to_csv(os.path.join(results_dir, "eval_full_train_metrics.csv"), index=False)
    df_test.to_csv(os.path.join(results_dir, "eval_full_test_metrics.csv"), index=False)

    # 输出统计报告
    print("\n==========================================================================")
    print("📊 [전체 크기 (145x689) Diffusion 물리 평가 최종 보고서]")
    print("==========================================================================")

    for name, df in [("Train Dataset", df_train), ("Test Dataset", df_test)]:
        print(f"【{name}】 (샘플 수: {len(df)})")
        print(f"  - 종합 상대 L2 오차 (Combined L2) :  {df['comb_l2 (%)'].mean():6.3f}% ± {df['comb_l2 (%)'].std():6.3f}%")
        print("  ------------------------------------------------------------------------")
        print(f"  - U 속도장 : L2 = {df['l2_u (%)'].mean():6.3f}% | RMSE = {df['rmse_u'].mean():.4f} m/s | MAE = {df['mae_u'].mean():.4f} m/s")
        print(f"  - V 속도장 : L2 = {df['l2_v (%)'].mean():6.3f}% | RMSE = {df['rmse_v'].mean():.4f} m/s | MAE = {df['mae_v'].mean():.4f} m/s")
        print(f"  - P 압력장 : L2 = {df['l2_p (%)'].mean():6.3f}% | RMSE = {df['rmse_p'].mean():.4f} Pa  | MAE = {df['mae_p'].mean():.4f} Pa")
        print("==========================================================================\n")

    # 提取表现最佳的测试样本进行精细化可视化
    best_idx = df_test["comb_l2 (%)"].idxmin()
    best_sample_info = df_test.loc[best_idx]

    print("🏆 ========================================================================")
    print("🏆 [테스트 데이터셋 내 최고 성능 (L2 지표 최상) 샘플 분석]")
    print("🏆 ========================================================================")
    print(f"  - 프레임 인덱스 (Frame Index)   : #{int(best_sample_info['sample_idx'])}")
    print(f"  - 물리 타임스텝 (Time Step t)  : {best_sample_info['time_step']:.1f}")
    print(f"  - ⭐ 종합 상대 L2 오차 (Best L2) : {best_sample_info['comb_l2 (%)']:.3f}%")
    print("  ------------------------------------------------------------------------")
    print(f"  - U 속도장 (U Velocity)  : L2 = {best_sample_info['l2_u (%)']:.3f}% | RMSE = {best_sample_info['rmse_u']:.4f} m/s | MAE = {best_sample_info['mae_u']:.4f} m/s")
    print(f"  - V 속도장 (V Velocity)  : L2 = {best_sample_info['l2_v (%)']:.3f}% | RMSE = {best_sample_info['rmse_v']:.4f} m/s | MAE = {best_sample_info['mae_v']:.4f} m/s")
    print(f"  - P 압력장 (Pressure)    : L2 = {best_sample_info['l2_p (%)']:.3f}% | RMSE = {best_sample_info['rmse_p']:.4f} Pa  | MAE = {best_sample_info['mae_p']:.4f} Pa")
    print("==========================================================================\n")

    plot_save_path = os.path.join(results_dir, f"contour_best_full_test_t{best_sample_info['time_step']:.0f}.png")

    plot_contour_comparison(
        grid_x_raw,
        grid_y_raw,
        test_gt[best_idx],
        test_pred[best_idx],
        best_sample_info['time_step'],
        plot_save_path,
    )