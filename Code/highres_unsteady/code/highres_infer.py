#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
자동 크기 적응형 (full / down4) C-grid DiffusionUNet 추론 및 평가 스크립트

기능: DDIM 50단계 빠른 샘플링 + 물리 역정규화 + 다차원 오차 통계 + 등치선 시각화
"""

import os
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.stats import gaussian_kde

from model_utils import DiffusionUNet

matplotlib.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'SimHei', 'PingFang SC',
    'Noto Sans CJK SC', 'WenQuanYi Zen Hei',
    'DejaVu Sans'
]
matplotlib.rcParams['axes.unicode_minus'] = False   # 음수 기호 정상 표시
matplotlib.rcParams['figure.max_open_warning'] = 0  # 다중 플롯 경고 억제

#==========================================
# 0. 설정 (Configuration & Switches)
#==========================================
DATA_SUFFIX = "down1"       # "full", "down4", "down1" 등
TARGET_LOSS = "_mse"        # "_mae", "_mse", "_pys" 등

#---- 지정 시간 단계 (t) 단독 계산 ----
TARGET_TIMESTEPS = [179]   # 예: [120.0, 150.0] 또는 [] 또는 None (모든 테스트셋)
EVAL_ALL = True            # True: 전체 테스트셋 평가, False: TARGET_TIMESTEPS만 처리

#---- 평가 및 시각화 제어 스위치 ----
EVAL_TRAIN = False          # True: 훈련셋도 함께 추론/평가, False: 테스트셋만 추론 (시간 절약)
ENABLE_PLOTS = False         # True: 시각화 이미지 생성 및 출력, False: 이미지 생성 안 함

#==========================================
# 1. DDIM 결정론적 빠른 샘플러 (DDIM Sampler)
#==========================================
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

        x0_pred = model(x_t, grid_x_b, grid_y_b, t, target_cond_batch)
        x0_pred = torch.clamp(x0_pred, -1.2, 1.2)

        if i == len(times) - 1:
            x_t = x0_pred
            break

        t_next_idx = times[i + 1]
        ab_t = alpha_bar[t_idx]
        ab_next = alpha_bar[t_next_idx]

        eps_pred = (x_t - torch.sqrt(ab_t) * x0_pred) / (torch.sqrt(1.0 - ab_t) + 1e-8)
        x_t = torch.sqrt(ab_next) * x0_pred + torch.sqrt(1.0 - ab_next) * eps_pred

    return x_t

#==========================================
# 2. 물리량 역정규화 및 다차원 오차 계산
#==========================================
def denormalize_batch(field_tensor, f_min, f_max):
    if isinstance(field_tensor, torch.Tensor):
        arr = field_tensor.detach().cpu().numpy()
    else:
        arr = np.array(field_tensor)

    f_min_np = f_min.cpu().numpy() if isinstance(f_min, torch.Tensor) else np.array(f_min)
    f_max_np = f_max.cpu().numpy() if isinstance(f_max, torch.Tensor) else np.array(f_max)

    phys_0_1 = (arr + 1.0) / 2.0
    f_min_c = f_min_np.reshape(1, 4, 1, 1)
    f_max_c = f_max_np.reshape(1, 4, 1, 1)
    return phys_0_1 * (f_max_c - f_min_c) + f_min_c

def calculate_metrics_single(gt_single, pred_single):
    eps = 1e-5
    ch_rho, ch_u, ch_v, ch_p = 0, 1, 2, 3

    # 원시 상대오차 (소수 형태)
    rel_u = np.linalg.norm(gt_single[ch_u] - pred_single[ch_u]) / (np.linalg.norm(gt_single[ch_u]) + eps)
    rel_v = np.linalg.norm(gt_single[ch_v] - pred_single[ch_v]) / (np.linalg.norm(gt_single[ch_v]) + eps)
    rel_p = np.linalg.norm(gt_single[ch_p] - pred_single[ch_p]) / (np.linalg.norm(gt_single[ch_p]) + eps)

    # 백분율 형태 (%)
    l2_u = rel_u * 100
    l2_v = rel_v * 100
    l2_p = rel_p * 100

    # 조합 상대오차 (3 가지 물리량 통합)
    u_p, u_t = pred_single[ch_u].flatten(), gt_single[ch_u].flatten()
    v_p, v_t = pred_single[ch_v].flatten(), gt_single[ch_v].flatten()
    p_p, p_t = pred_single[ch_p].flatten(), gt_single[ch_p].flatten()
    res_sq = np.sum((u_p - u_t)**2) + np.sum((v_p - v_t)**2) + np.sum((p_p - p_t)**2)
    den_sq = np.sum(u_t**2) + np.sum(v_t**2) + np.sum(p_t**2)
    comb_rel = np.sqrt(res_sq / (den_sq + eps))          # 소수
    comb_l2 = comb_rel * 100                             # 백분율 (%)

    # RMSE
    rmse_u = np.sqrt(np.mean((gt_single[ch_u] - pred_single[ch_u])**2))
    rmse_v = np.sqrt(np.mean((gt_single[ch_v] - pred_single[ch_v])**2))
    rmse_p = np.sqrt(np.mean((gt_single[ch_p] - pred_single[ch_p])**2))

    # MAE
    mae_u = np.mean(np.abs(gt_single[ch_u] - pred_single[ch_u]))
    mae_v = np.mean(np.abs(gt_single[ch_v] - pred_single[ch_v]))
    mae_p = np.mean(np.abs(gt_single[ch_p] - pred_single[ch_p]))

    # NRMSE (CFDLib relative_error)
    def nrmse(gt, pred):
        numerator = np.mean((pred - gt) ** 2)
        denominator = dst(np.mean((gt - np.mean(gt)) ** 2))
        return np.sqrt(numerator / (denominator + eps))

    nrmse_u = nrmse(gt_single[ch_u], pred_single[ch_u])
    nrmse_v = nrmse(gt_single[ch_v], pred_single[ch_v])
    nrmse_p = nrmse(gt_single[ch_p], pred_single[ch_p])

    # MSE
    mse_u = np.mean((gt_single[ch_u] - pred_single[ch_u]) ** 2)
    mse_v = np.mean((gt_single[ch_v] - pred_single[ch_v]) ** 2)
    mse_p = np.mean((gt_single[ch_p] - pred_single[ch_p]) ** 2)

    return {
        "comb_l2": comb_l2, "comb_rel": comb_rel,
        "l2_u": l2_u, "l2_v": l2_v, "l2_p": l2_p,
        "rel_u": rel_u, "rel_v": rel_v, "rel_p": rel_p,
        "rmse_u": rmse_u, "rmse_v": rmse_v, "rmse_p": rmse_p,
        "mae_u": mae_u, "mae_v": mae_v, "mae_p": mae_p,
        "nrmse_u": nrmse_u, "nrmse_v": nrmse_v, "nrmse_p": nrmse_p,
        "mse_u": mse_u, "mse_v": mse_v, "mse_p": mse_p
    }

#==========================================
# 3. 데이터셋 평가 및 추론 루프
#==========================================
def evaluate_selected_samples(dataset_name, data_npz, model, device, grid_x_tensor, grid_y_tensor,
                              f_min, f_max, l_min, l_max, sample_indices, ddim_steps=50, batch_size=4):
    print(f"\n🚀 선택된 {len(sample_indices)}개 샘플 추론 시작: 【{dataset_name}】")

    if len(sample_indices) == 0:
        print("⚠️ 샘플 인덱스가 비어 있습니다.")
        return None, None, None

    x_data = data_npz['x']
    y_data = data_npz['y']
    results = []
    all_gt_phys = []
    all_pred_phys = []

    for i in tqdm(range(0, len(sample_indices), batch_size), desc=f"Inferencing {dataset_name}"):
        batch_indices = sample_indices[i:i+batch_size]
        batch_x_norm = x_data[batch_indices]
        batch_y_norm = y_data[batch_indices]

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

        for j, idx in enumerate(batch_indices):
            # DeprecationWarning 방지를 위한 안전한 스칼라 추출
            y_val = batch_y_norm[j].item() if hasattr(batch_y_norm[j], 'item') else float(batch_y_norm[j])
            time_val = float(y_val * (l_max - l_min + 1e-8) + l_min)

            metrics = calculate_metrics_single(gt_phys[j], pred_phys[j])

            entry = {
                "sample_idx": idx,
                "time_step": time_val,
                "comb_l2 (%)": metrics["comb_l2"],
                "comb_rel": metrics["comb_rel"],
                "l2_u (%)": metrics["l2_u"], "l2_v (%)": metrics["l2_v"], "l2_p (%)": metrics["l2_p"],
                "rel_u": metrics["rel_u"], "rel_v": metrics["rel_v"], "rel_p": metrics["rel_p"],
                "rmse_u": metrics["rmse_u"], "rmse_v": metrics["rmse_v"], "rmse_p": metrics["rmse_p"],
                "mae_u": metrics["mae_u"], "mae_v": metrics["mae_v"], "mae_p": metrics["mae_p"],
                "nrmse_u": metrics["nrmse_u"], "nrmse_v": metrics["nrmse_v"], "nrmse_p": metrics["nrmse_p"],
                "mse_u": metrics["mse_u"], "mse_v": metrics["mse_v"], "mse_p": metrics["mse_p"]
            }
            results.append(entry)

    df = pd.DataFrame(results)
    all_gt_phys = np.concatenate(all_gt_phys, axis=0)
    all_pred_phys = np.concatenate(all_pred_phys, axis=0)

    print(f"✅ {dataset_name} 선택 샘플 추론 완료!")
    return df, all_gt_phys, all_pred_phys

#==========================================
# 4. 시각화 함수들 (viridis 컬러맵 적용)
#==========================================
def plot_contour_comparison(grid_x_raw, grid_y_raw, gt_sample, pred_sample, time_val, save_path):
    ch_u, ch_v, ch_p = 1, 2, 3

    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True, sharey=True)
    fig.suptitle(
        f"Grid {grid_x_raw.shape} 고밀도 등치선 (Time = {time_val:.1f})",
        fontsize=14,
        fontweight="bold",
    )

    cmap_field = "viridis"
    cmap_err = "viridis"

    channels = [
        ("U 속도", ch_u, "m/s"),
        ("V 속도", ch_v, "m/s"),
        ("압력", ch_p, "Pa"),
    ]

    for row, (name, ch_idx, unit) in enumerate(channels):
        gt = gt_sample[ch_idx]
        pred = pred_sample[ch_idx]
        abs_err = np.abs(gt - pred)

        val_min = min(gt.min(), pred.min())
        val_max = max(gt.max(), pred.max())
        levels_dense = np.linspace(val_min, val_max, 35)

        # Ground Truth
        ax_gt = axes[row, 0]
        c0 = ax_gt.pcolormesh(grid_x_raw, grid_y_raw, gt, shading="gouraud", cmap=cmap_field)
        ax_gt.contour(grid_x_raw, grid_y_raw, gt, levels=levels_dense, colors="black", linewidths=0.35, alpha=0.55)
        ax_gt.contour(grid_x_raw, grid_y_raw, gt, levels=levels_dense[::5], colors="black", linewidths=0.7, alpha=0.85)
        ax_gt.set_title(f"Ground Truth - {name}")
        fig.colorbar(c0, ax=ax_gt, label=unit)

        # Prediction
        ax_pred = axes[row, 1]
        c1 = ax_pred.pcolormesh(grid_x_raw, grid_y_raw, pred, shading="gouraud", cmap=cmap_field)
        ax_pred.contour(grid_x_raw, grid_y_raw, pred, levels=levels_dense, colors="black", linewidths=0.35, alpha=0.55)
        ax_pred.contour(grid_x_raw, grid_y_raw, pred, levels=levels_dense[::5], colors="black", linewidths=0.7, alpha=0.85)
        ax_pred.set_title(f"DDIM Prediction - {name}")
        fig.colorbar(c1, ax=ax_pred, label=unit)

        # Error
        ax_err = axes[row, 2]
        c2 = ax_err.pcolormesh(grid_x_raw, grid_y_raw, abs_err, shading="gouraud", cmap=cmap_err)
        err_levels = np.linspace(0, abs_err.max() + 1e-8, 20)
        ax_err.contour(grid_x_raw, grid_y_raw, abs_err, levels=err_levels, colors="white", linewidths=0.3, alpha=0.6)

        mean_err = np.mean(abs_err)
        max_err = np.max(abs_err)
        ax_err.set_title(f"절 đối 오차 - {name} (MAE: {mean_err:.4f}, Max: {max_err:.4f})")
        fig.colorbar(c2, ax=ax_err, label=unit)

    for ax in axes.flatten():
        ax.plot(grid_x_raw[0, :], grid_y_raw[0, :], "k-", linewidth=1.2, zorder=10)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"🖼️ 고밀도 등치선이 포함된 정밀 등고선도가 저장되었습니다: {save_path}")
    plt.show()

def plot_colormap_only(grid_x_raw, grid_y_raw, gt_sample, pred_sample, time_val, save_path):
    ch_u, ch_v, ch_p = 1, 2, 3
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True, sharey=True)
    fig.suptitle(
        f"GT vs Prediction (컬러 전용, 등치선 없음) – Time = {time_val:.1f}",
        fontsize=14,
        fontweight="bold",
    )

    cmap = "viridis"

    channels = [
        ("U 속도", ch_u, "m/s"),
        ("V 속도", ch_v, "m/s"),
        ("압력", ch_p, "Pa"),
    ]

    for row, (name, ch_idx, unit) in enumerate(channels):
        gt = gt_sample[ch_idx]
        pred = pred_sample[ch_idx]

        ax_gt = axes[row, 0]
        im_gt = ax_gt.pcolormesh(grid_x_raw, grid_y_raw, gt, shading="gouraud", cmap=cmap)
        ax_gt.set_title(f"GT - {name}")
        fig.colorbar(im_gt, ax=ax_gt, label=unit)

        ax_pred = axes[row, 1]
        im_pred = ax_pred.pcolormesh(grid_x_raw, grid_y_raw, pred, shading="gouraud", cmap=cmap)
        ax_pred.set_title(f"Pred - {name}")
        fig.colorbar(im_pred, ax=ax_pred, label=unit)

    for ax in axes.flatten():
        ax.plot(grid_x_raw[0, :], grid_y_raw[0, :], "k-", linewidth=1.5, zorder=10)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"🖼️ 등치선 없는 컬러맵 시각화 저장 완료: {save_path}")
    plt.show()

#==========================================
# 4b. 오차 통계 시각화 (R² / RMSE / KDE + Hist)
#==========================================
def plot_error_statistics(gt_phys, pred_phys, save_path, ch_names=("U", "V", "p")):
    """
    Beautified error summary panel:
      (a) R² bar with 0.9 reference line
      (b) RMSE bar (log scale)
      (c) Normalized error distribution:
            top    -> KDE curves (linear axis)
            bottom -> histogram (log axis), sharex with top
    gt_phys, pred_phys : (N, 4, H, W) numpy arrays
    Uses channels 1,2,3 for U, V, p (channel 0 is density rho).
    """
    from matplotlib import gridspec

    CH_COLORS  = ["#2E86AB", "#5CB85C", "#D9534F"]
    ch_indices = [1, 2, 3]

    #---- per-channel metrics ----
    metrics = {}
    for c_out, ch in enumerate(ch_indices):
        name = ch_names[c_out]
        p = pred_phys[:, ch]
        g = gt_phys[:, ch]

        mse  = float(np.mean((p - g) ** 2))
        rmse = float(np.sqrt(mse))
        mae  = float(np.mean(np.abs(p - g)))
        maxe = float(np.max(np.abs(p - g)))
        rel  = float(np.linalg.norm(p - g) / (np.linalg.norm(g) + 1e-8))
        ss_res = float(np.sum((p - g) ** 2))
        ss_tot = float(np.sum((g - g.mean()) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)

        metrics[name] = dict(mse=mse, rmse=rmse, mae=mae,
                             maxe=maxe, rel=rel, r2=r2)

    #---- typography ----
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig_stat = plt.figure(figsize=(20, 6))
    gs = gridspec.GridSpec(
        1, 3, figure=fig_stat, wspace=0.30,
        left=0.06, right=0.98, top=0.86, bottom=0.13,
    )

    #---- (a) R² bar ----
    ax = fig_stat.add_subplot(gs[0, 0])
    r2_vals = [metrics[n]['r2'] for n in ch_names]
    bars = ax.bar(ch_names, r2_vals, color=CH_COLORS, width=0.55,
                  edgecolor='black', linewidth=1.2, zorder=3)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R²")
    ax.set_title("결정 계수 (R²)",
                 fontweight='bold', pad=12)
    ax.axhline(0.9, color='gray', linestyle='--', linewidth=1.2,
               alpha=0.7, zorder=2, label='R² = 0.9')
    for bar, v in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.025, f'{v:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.grid(True, axis='y', linestyle=':', alpha=0.5, zorder=0)
    ax.legend(loc='lower right', frameon=False)

    #---- (b) RMSE bar (log) ----
    ax = fig_stat.add_subplot(gs[0, 1])
    rmse_vals = [metrics[n]['rmse'] for n in ch_names]
    bars = ax.bar(ch_names, rmse_vals, color=CH_COLORS, width=0.55,
                  edgecolor='black', linewidth=1.2, zorder=3)
    ax.set_yscale('log')
    ax.set_ylabel("RMSE (log scale)")
    ax.set_title("평균 제곱근 오차",
                 fontweight='bold', pad=12)
    for bar, v in zip(bars, rmse_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v * 1.15, f'{v:.2e}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.grid(True, axis='y', linestyle=':', alpha=0.5,
            which='both', zorder=0)

    #---- (c) KDE (top) + Hist (bottom), sharex ----
    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[0, 2],
        height_ratios=[1.0, 1.15], hspace=0.12,
    )
    ax_kde  = fig_stat.add_subplot(gs_right[0])
    ax_hist = fig_stat.add_subplot(gs_right[1], sharex=ax_kde)

    #normalized errors (clipped to [-4, 4])
    norm_errors = {}
    for c_out, ch in enumerate(ch_indices):
        name = ch_names[c_out]
        err = (pred_phys[:, ch] - gt_phys[:, ch]).ravel()
        sigma = gt_phys[:, ch].std() + 1e-8
        norm_errors[name] = np.clip(err / sigma, -4, 4)

    xs = np.linspace(-4, 4, 400)
    for c_out, name in enumerate(ch_names):
        err_norm = norm_errors[name]
        #KDE is O(N^2); subsample if huge
        if len(err_norm) > 20000:
            rng = np.random.default_rng(42)
            err_sub = rng.choice(err_norm, size=20000, replace=False)
        else:
            err_sub = err_norm
        kde = gaussian_kde(err_sub, bw_method=0.15)
        ax_kde.plot(xs, kde(xs), color=CH_COLORS[c_out],
                    linewidth=2.2, label=name, zorder=3)

    ax_kde.set_ylabel("KDE density")
    ax_kde.set_title("정규화된 오차 분포",
                     fontweight='bold', pad=12)
    ax_kde.legend(frameon=False, loc='upper right', ncol=3)
    ax_kde.grid(True, linestyle=':', alpha=0.5, zorder=0)
    ax_kde.set_xlim(-4, 4)
    plt.setp(ax_kde.get_xticklabels(), visible=False)

    for c_out, name in enumerate(ch_names):
        ax_hist.hist(norm_errors[name], bins=120, alpha=0.55,
                     color=CH_COLORS[c_out], density=True,
                     label=name, zorder=2, range=(-4, 4))

    ax_hist.set_yscale('log')
    ax_hist.set_xlabel("정규화 오차  (err / σ_true)")
    ax_hist.set_ylabel("히스토그램 밀도 (log scale)")
    ax_hist.set_xlim(-4, 4)
    ax_hist.grid(True, linestyle=':', alpha=0.5, which='both', zorder=0)
    ax_hist.legend(frameon=False, loc='upper right', ncol=3)

    fig_stat.suptitle("Diffusion C-그리드 — 오차 요약",
                      fontsize=15, fontweight='bold', y=0.97)

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig_stat)
    print(f"🖼️ 오차 통계 이미지 저장 완료: {save_path}")

    return metrics   # 반환하여 추가 활용 가능

#==========================================
# 5. 보조 함수: 시간 단계로 인덱스 찾기
#==========================================
def find_samples_by_time(data_npz, target_times, l_min, l_max, tol=1e-3):
    if target_times is None or len(target_times) == 0:
        return list(range(len(data_npz['y'])))
    y_norm = data_npz['y']
    if y_norm.ndim > 1:
        y_norm = y_norm.reshape(-1)
    l_min = float(np.asarray(l_min).ravel()[0])
    l_max = float(np.asarray(l_max).ravel()[0])

    indices = []
    for t in target_times:
        t_norm = (t - l_min) / (l_max - l_min + 1e-8)
        diff = np.abs(y_norm - t_norm)
        idx = np.argmin(diff)
        if diff[idx] < tol:
            indices.append(idx)
        else:
            print(f"⚠️ 시간 {t}에 해당하는 샘플을 찾을 수 없습니다 (가장 가까운 값: {y_norm[idx]*(l_max-l_min)+l_min:.2f})")
    return sorted(set(indices))

#==========================================
# 6. 메인 실행 진입점
#==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🌟 {DATA_SUFFIX} 데이터셋 Diffusion 추론 및 평가 프로세스 시작 | 장치: {device}")

    DDIM_STEPS = 50
    BATCH_SIZE = 4

    current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
    results_dir = os.path.abspath(os.path.join(current_dir, "../Results"))

    #--- 1) 가중치 파일 ---
    weights_candidates = [
        os.path.join(results_dir, f"airfoil_diffusion_cgrid_{DATA_SUFFIX}_final{TARGET_LOSS}.pth"),
        os.path.join(results_dir, f"airfoil_diffusion_cgrid_{DATA_SUFFIX}_ep3000.pth"),
        os.path.join(results_dir, f"airfoil_diffusion_cgrid_{DATA_SUFFIX}_ep2500.pth"),
        os.path.join(results_dir, f"airfoil_diffusion_cgrid_{DATA_SUFFIX}_ep2000.pth"),
    ]
    weights_path = next((w for w in weights_candidates if os.path.exists(w)), None)
    if weights_path is None:
        raise FileNotFoundError(f"❌ 가중치 파일을 찾을 수 없습니다 (접미사: {DATA_SUFFIX})")

    #--- 2) 정규화 계수 및 데이터 파일 ---
    norm_path = os.path.join(results_dir, f"normalization_factors_{DATA_SUFFIX}.npz")
    train_data_path = os.path.join(results_dir, f"Diffusion_airfoil_unsteady_{DATA_SUFFIX}_train.npz")
    test_data_path = os.path.join(results_dir, f"Diffusion_airfoil_unsteady_{DATA_SUFFIX}_test.npz")

    if not os.path.exists(norm_path) or not os.path.exists(test_data_path):
        raise FileNotFoundError(f"❌ 필수 데이터 파일이 누락되었습니다.")

    #--- 모델 로드 ---
    model = DiffusionUNet(flow_ch=4, coord_ch=2, cond_dim=128, base_ch=48).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    print(f"✅ 가중치 모델 로드 성공: {weights_path}")

    #--- 데이터 로드 ---
    test_data = np.load(test_data_path)
    train_data = np.load(train_data_path) if (EVAL_TRAIN and os.path.exists(train_data_path)) else None
    norm_factors = np.load(norm_path)

    f_min, f_max = norm_factors["fields_min"], norm_factors["fields_max"]
    l_min, l_max = norm_factors["label_min"], norm_factors["label_max"]

    l_min = float(np.asarray(l_min).ravel()[0])
    l_max = float(np.asarray(l_max).ravel()[0])

    # 그리드 좌표 (훈련/테스트 공통 사용)
    grid_x_raw = test_data["grid_x"] if "grid_x" in test_data else np.load(train_data_path)["grid_x"]
    grid_y_raw = test_data["grid_y"] if "grid_y" in test_data else np.load(train_data_path)["grid_y"]

    grid_x_tensor = torch.tensor(grid_x_raw, dtype=torch.float32, device=device)
    grid_y_tensor = torch.tensor(grid_y_raw, dtype=torch.float32, device=device)
    grid_x_tensor = 2.0 * (grid_x_tensor - grid_x_tensor.min()) / (grid_x_tensor.max() - grid_x_tensor.min() + 1e-8) - 1.0
    grid_y_tensor = 2.0 * (grid_y_tensor - grid_y_tensor.min()) / (grid_y_tensor.max() - grid_y_tensor.min() + 1e-8) - 1.0
    if grid_x_tensor.dim() == 2:
        grid_x_tensor = grid_x_tensor.unsqueeze(0).unsqueeze(0)
        grid_y_tensor = grid_y_tensor.unsqueeze(0).unsqueeze(0)

    #==========================================
    # 분기 1: 전체 평가 (EVAL_ALL = True)
    #==========================================
    if EVAL_ALL or TARGET_TIMESTEPS is None or len(TARGET_TIMESTEPS) == 0:
        print("🔍 전체 테스트셋 평가를 수행합니다.")

        #1) 테스트셋 추론
        df_test, test_gt, test_pred = evaluate_selected_samples(
            "Test Dataset", test_data, model, device,
            grid_x_tensor, grid_y_tensor, f_min, f_max, l_min, l_max,
            sample_indices=list(range(len(test_data['x']))),
            ddim_steps=DDIM_STEPS, batch_size=BATCH_SIZE
        )

        #2) 훈련셋 조건부 추론 (EVAL_TRAIN = True 인 경우에만)
        df_train = None
        if EVAL_TRAIN and train_data is not None:
            print("🔍 훈련셋 평가를 수행합니다. (EVAL_TRAIN = True)")
            df_train, train_gt, train_pred = evaluate_selected_samples(
                "Train Dataset", train_data, model, device,
                grid_x_tensor, grid_y_tensor, f_min, f_max, l_min, l_max,
                sample_indices=list(range(len(train_data['x']))),
                ddim_steps=DDIM_STEPS, batch_size=BATCH_SIZE
            )
        else:
            print("⚡ 훈련셋 평가를 건너뜁니다. (EVAL_TRAIN = False)")

        #3) 전체 통계 보고서 출력
        eval_list = [("Test Dataset", df_test)]
        if df_train is not None:
            eval_list.insert(0, ("Train Dataset", df_train))

        print("\n==========================================================================")
        print(f"📊 [{DATA_SUFFIX} 데이터셋 Diffusion 물리 평가 최종 보고서]")
        print("==========================================================================")

        for name, df in eval_list:
            print(f"【{name}】 (샘플 수: {len(df)})")
            print(f"  - 종합 상대 L2 오차 (Combined L2) :  {df['comb_l2 (%)'].mean():6.3f}% ± {df['comb_l2 (%)'].std():6.3f}%")
            print("  ------------------------------------------------------------------------")
            print(f"  - U 속도장 : L2 = {df['l2_u (%)'].mean():6.3f}% | RMSE = {df['rmse_u'].mean():.4f} m/s | MAE = {df['mae_u'].mean():.4f} m/s")
            print(f"  - V 속도장 : L2 = {df['l2_v (%)'].mean():6.3f}% | RMSE = {df['rmse_v'].mean():.4f} m/s | MAE = {df['mae_v'].mean():.4f} m/s")
            print(f"  - P 압력장 : L2 = {df['l2_p (%)'].mean():6.3f}% | RMSE = {df['rmse_p'].mean():.4f} Pa  | MAE = {df['mae_p'].mean():.4f} Pa")
            print(f"  - NRMSE (CFDLib) : U = {df['nrmse_u'].mean():.6f}, V = {df['nrmse_v'].mean():.6f}, P = {df['nrmse_p'].mean():.6f}")
            print(f"  - MSE   (CFDLib) : U = {df['mse_u'].mean():.6f}, V = {df['mse_v'].mean():.6f}, P = {df['mse_p'].mean():.6f}")
            print("--------------------------------------------------------------------------")

        #4) 최우수 성능 샘플 추출 및 시각화 (테스트셋 기준)
        best_idx = df_test["comb_l2 (%)"].idxmin()
        best_sample_info = df_test.loc[best_idx]

        print("\n🏆 ========================================================================")
        print("🏆 [테스트 데이터셋 내 최우수 성능 (L2 지표 최상) 샘플 분석]")
        print("🏆 ========================================================================")
        print(f"  - 프레임 인덱스 (Frame Index)   : #{int(best_sample_info['sample_idx'])}")
        print(f"  - 물리 타임스텝 (Time Step t)  : {best_sample_info['time_step']:.1f}")
        print(f"  - ⭐ 종합 상대 L2 오차 (Best L2) : {best_sample_info['comb_l2 (%)']:.3f}%")
        print("==========================================================================\n")

        if ENABLE_PLOTS:
            plot_save_contour = os.path.join(results_dir, f"contour_best_{DATA_SUFFIX}_t{best_sample_info['time_step']:.0f}.png")
            plot_contour_comparison(grid_x_raw, grid_y_raw, test_gt[best_idx], test_pred[best_idx],
                                    best_sample_info['time_step'], plot_save_contour)
            plot_save_nocolor = os.path.join(results_dir, f"colormap_best_{DATA_SUFFIX}_t{best_sample_info['time_step']:.0f}_nocolor.png")
            plot_colormap_only(grid_x_raw, grid_y_raw, test_gt[best_idx], test_pred[best_idx],
                               best_sample_info['time_step'], plot_save_nocolor)

            #⭐ NEW: 전체 테스트셋 기준 오차 통계 패널
            err_stats_path = os.path.join(results_dir, f"error_statistics_{DATA_SUFFIX}.png")
            plot_error_statistics(test_gt, test_pred, err_stats_path, ch_names=("U", "V", "p"))

    #==========================================
    # 분기 2: 지정 시간 단계 평가 (EVAL_ALL = False)
    #==========================================
    else:
        print(f"🔍 지정된 시간 단계 {TARGET_TIMESTEPS} 에 대한 샘플을 처리합니다.")

        #1) 테스트셋 특정 시간 단계 평가
        test_indices = find_samples_by_time(test_data, TARGET_TIMESTEPS, l_min, l_max, tol=1e-3)
        if len(test_indices) > 0:
            print(f"✅ 테스트셋 발견된 샘플 인덱스: {test_indices}")
            df_test, test_gt, test_pred = evaluate_selected_samples(
                "Test Dataset (Selected)", test_data, model, device,
                grid_x_tensor, grid_y_tensor, f_min, f_max, l_min, l_max,
                sample_indices=test_indices,
                ddim_steps=DDIM_STEPS, batch_size=BATCH_SIZE
            )
            test_csv = os.path.join(results_dir, f"eval_{DATA_SUFFIX}_test_selected_times.csv")
            df_test.to_csv(test_csv, index=False)

            #⭐ NEW: 선택된 시간 단계 기준 오차 통계 패널
            if ENABLE_PLOTS:
                err_stats_path = os.path.join(results_dir, f"error_statistics_{DATA_SUFFIX}_selected_times.png")
                plot_error_statistics(test_gt, test_pred, err_stats_path, ch_names=("U", "V", "p"))
            if ENABLE_PLOTS and len(test_indices) > 0:
                row = df_test.iloc[0]
                sample_pos = test_indices.index(int(row['sample_idx']))
                plot_contour_comparison(grid_x_raw, grid_y_raw, test_gt[sample_pos], test_pred[sample_pos],
                                        row['time_step'], os.path.join(results_dir, f"contour_test_{DATA_SUFFIX}_t{row['time_step']:.0f}.png"))
        else:
            print("⚠️ 테스트셋에서 지정된 시간 단계에 해당하는 샘플이 없습니다.")

        #2) 훈련셋 특정 시간 단계 평가 (EVAL_TRAIN = True 인 경우에만)
        if EVAL_TRAIN and train_data is not None:
            train_indices = find_samples_by_time(train_data, TARGET_TIMESTEPS, l_min, l_max, tol=1e-3)
            if len(train_indices) > 0:
                print(f"✅ 훈련셋 발견된 샘플 인덱스: {train_indices}")
                df_train, train_gt, train_pred = evaluate_selected_samples(
                    "Train Dataset (Selected)", train_data, model, device,
                    grid_x_tensor, grid_y_tensor, f_min, f_max, l_min, l_max,
                    sample_indices=train_indices,
                    ddim_steps=DDIM_STEPS, batch_size=BATCH_SIZE
                )
                train_csv = os.path.join(results_dir, f"eval_{DATA_SUFFIX}_train_selected_times.csv")
                df_train.to_csv(train_csv, index=False)

                #⭐ NEW: 훈련셋 선택 시간 단계 기준 오차 통계
                if ENABLE_PLOTS:
                    err_stats_path = os.path.join(results_dir, f"error_statistics_{DATA_SUFFIX}_train_selected_times.png")
                    plot_error_statistics(train_gt, train_pred, err_stats_path, ch_names=("U", "V", "p"))
                if ENABLE_PLOTS and len(train_indices) > 0:
                    row = df_train.iloc[0]
                    sample_pos = train_indices.index(int(row['sample_idx']))
                    plot_contour_comparison(grid_x_raw, grid_y_raw, train_gt[sample_pos], train_pred[sample_pos],
                                            row['time_step'], os.path.join(results_dir, f"contour_train_{DATA_SUFFIX}_t{row['time_step']:.0f}.png"))
            else:
                print("⚠️ 훈련셋에서 지정된 시간 단계에 해당하는 샘플이 없습니다.")
        else:
            print("⚡ 훈련셋 추론을 건너뜁니다. (EVAL_TRAIN = False)")

