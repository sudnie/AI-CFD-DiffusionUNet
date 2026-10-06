#!/usr/bin/python3#-*- coding: utf-8 -*-
"""
自动尺寸适应型 C-grid DiffusionUNet 推理与物理评估脚本
功能：DDIM 采样 + 逆归一化 + 雅可比变换(涡流) + 高斯平滑 + Cp/Cd/Cl
      绝对误差 / 相对误差 / 涡核区域误差 / L2 误差
"""

import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from model_utils import DiffusionUNet

matplotlib.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'SimHei', 'PingFang SC',
    'Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'DejaVu Sans'
]
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.max_open_warning'] = 0

#==========================================#0. 配置项#==========================================DATA_SUFFIX = "down1"
TARGET_LOSS = "_mse"
TARGET_TIMESTEPS = [179]
EVAL_ALL = True

ENABLE_PLOTS = True
SAVE_METRICS_CSV = False

#---- 涡流分析配置 ----VORTEX_METRIC = "vorticity"      # "q_criterion" 或 "vorticity"
VORTEX_SMOOTH_SIGMA = 0.8          # 求导前的高斯平滑 sigma (0.5~1.0)
VORTEX_CORE_PERCENTILE = 75        # 涡核判定分位数 (GT 中 Q>75% 视为涡核)

#---- Cp / Cd / Cl 积分设置 ----AIRFOIL_SURFACE_INDEX = 0
AIRFOIL_CHORD_GUESS = 1.0
AIRFOIL_Y_TOL = 0.30
CHORD_LENGTH_OVERRIDE = None


#==========================================#1. DDIM 采样器#==========================================@torch.no_grad()
def sample_ddim_batch(model, device, grid_x, grid_y, target_cond_batch,
                      total_timesteps=1000, ddim_steps=50):
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


#==========================================#2. 物理量逆归一化#==========================================def denormalize_batch(field_tensor, f_min, f_max):
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


#==========================================#3. 曲线网格雅可比映射 + 高斯平滑#==========================================def compute_vorticity_and_q_field_jacobian(phys_field, grid_x, grid_y, smooth_sigma=0.8):
    """
    雅可比坐标变换计算 du/dx, du/dy, dv/dx, dv/dy 及涡量/Q准则。
    求导前对 u, v 做高斯平滑以抑制网格尺度噪声。
    phys_field: [4, H, W] (0: rho, 1: u, 2: v, 3: p)
    """
    u = phys_field[1].copy()
    v = phys_field[2].copy()

    #平滑抑制数值噪声    if smooth_sigma > 0:
        u = gaussian_filter(u, sigma=smooth_sigma)
        v = gaussian_filter(v, sigma=smooth_sigma)

    dx_deta, dx_dxi = np.gradient(grid_x, edge_order=2)
    dy_deta, dy_dxi = np.gradient(grid_y, edge_order=2)
    du_deta, du_dxi = np.gradient(u, edge_order=2)
    dv_deta, dv_dxi = np.gradient(v, edge_order=2)

    J = dx_dxi * dy_deta - dx_deta * dy_dxi
    J_safe = np.where(np.abs(J) < 1e-6, 1e-6, J)
    J_inv = np.clip(1.0 / J_safe, -1e4, 1e4)

    du_dx = (du_dxi * dy_deta - du_deta * dy_dxi) * J_inv
    du_dy = (du_deta * dx_dxi - du_dxi * dx_deta) * J_inv
    dv_dx = (dv_dxi * dy_deta - dv_deta * dy_dxi) * J_inv
    dv_dy = (dv_deta * dx_dxi - dv_dxi * dx_deta) * J_inv

    vorticity_z = dv_dx - du_dy

    S_11 = du_dx
    S_22 = dv_dy
    S_12 = 0.5 * (du_dy + dv_dx)
    Omega_12 = 0.5 * (du_dy - dv_dx)
    norm_Omega_sq = 2.0 * (Omega_12 ** 2)
    norm_S_sq = (S_11 ** 2) + (S_22 ** 2) + 2.0 * (S_12 ** 2)
    q_criterion = 0.5 * (norm_Omega_sq - norm_S_sq)

    return vorticity_z, q_criterion


#==========================================#4. Cp / Cd / Cl 적분#==========================================def estimate_freestream_conditions(gt_phys):
    p = gt_phys[3]; u = gt_phys[1]; v = gt_phys[2]; rho = gt_phys[0]
    edge_mask = np.zeros_like(p, dtype=bool)
    edge_mask[-1, :] = True
    edge_mask[1:, 0] = True
    edge_mask[1:, -1] = True

    p_inf = float(np.median(p[edge_mask]))
    u_inf = float(np.median(u[edge_mask]))
    v_inf = float(np.median(v[edge_mask]))
    rho_inf = float(np.median(rho[edge_mask]))
    V_inf = float(np.sqrt(u_inf ** 2 + v_inf ** 2))
    return p_inf, rho_inf, V_inf


def compute_cp_field(pressure, p_ref, q_ref):
    return (pressure - p_ref) / (q_ref + 1e-10)


def compute_cd_pressure_green(x_surf, y_surf, cp_surf, chord):
    x = np.append(x_surf, x_surf[0]); y = np.append(y_surf, y_surf[0]); cp = np.append(cp_surf, cp_surf[0])
    is_ccw = np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) > 0
    dy = np.diff(y); cp_mid = 0.5 * (cp[:-1] + cp[1:])
    integral = np.sum(cp_mid * dy)
    return float((-integral if is_ccw else integral) / (chord + 1e-10))


def compute_cl_pressure_green(x_surf, y_surf, cp_surf, chord):
    x = np.append(x_surf, x_surf[0]); y = np.append(y_surf, y_surf[0]); cp = np.append(cp_surf, cp_surf[0])
    is_ccw = np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) > 0
    dx = np.diff(x); cp_mid = 0.5 * (cp[:-1] + cp[1:])
    integral = np.sum(cp_mid * dx)
    return float((integral if is_ccw else -integral) / (chord + 1e-10))


def find_airfoil_segment(grid_x_raw, grid_y_raw, airfoil_idx=0,
                         chord_guess=1.0, y_tol=0.30):
    x = grid_x_raw[airfoil_idx, :]; y = grid_y_raw[airfoil_idx, :]
    x_lo = -0.05 * chord_guess; x_hi = 0.975 * chord_guess
    mask = (x >= x_lo) & (x <= x_hi) & (np.abs(y) < y_tol * chord_guess)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise RuntimeError(f"row {airfoil_idx} 无法定位机翼节点段")
    breaks = np.where(np.diff(idx) > 1)[0]
    if len(breaks) == 0:
        i0, i1 = int(idx[0]), int(idx[-1]) + 1
    else:
        segs = np.split(idx, breaks + 1)
        longest = max(segs, key=len)
        i0, i1 = int(longest[0]), int(longest[-1]) + 1
    return i0, i1


def compute_cp_cd_cl(gt_phys, pred_phys, grid_x_raw, grid_y_raw,
                     airfoil_idx=AIRFOIL_SURFACE_INDEX,
                     chord_guess=AIRFOIL_CHORD_GUESS,
                     y_tol=AIRFOIL_Y_TOL,
                     chord_override=CHORD_LENGTH_OVERRIDE):
    p_inf, rho_inf, V_inf = estimate_freestream_conditions(gt_phys)
    q_inf = 0.5 * rho_inf * V_inf ** 2

    cp_gt_field = compute_cp_field(gt_phys[3], p_inf, q_inf)
    cp_pred_field = compute_cp_field(pred_phys[3], p_inf, q_inf)

    i0, i1 = find_airfoil_segment(grid_x_raw, grid_y_raw, airfoil_idx=airfoil_idx,
                                  chord_guess=chord_guess, y_tol=y_tol)

    x_surf = grid_x_raw[airfoil_idx, i0:i1]; y_surf = grid_y_raw[airfoil_idx, i0:i1]
    cp_gt_surf = cp_gt_field[airfoil_idx, i0:i1]
    cp_pred_surf = cp_pred_field[airfoil_idx, i0:i1]

    chord_eff = float(np.max(x_surf) - np.min(x_surf)) if chord_override is None else float(chord_override)

    cd_gt = compute_cd_pressure_green(x_surf, y_surf, cp_gt_surf, chord_eff)
    cd_pred = compute_cd_pressure_green(x_surf, y_surf, cp_pred_surf, chord_eff)
    cl_gt = compute_cl_pressure_green(x_surf, y_surf, cp_gt_surf, chord_eff)
    cl_pred = compute_cl_pressure_green(x_surf, y_surf, cp_pred_surf, chord_eff)

    vort_gt, q_gt = compute_vorticity_and_q_field_jacobian(
        gt_phys, grid_x_raw, grid_y_raw, smooth_sigma=VORTEX_SMOOTH_SIGMA)
    vort_pred, q_pred = compute_vorticity_and_q_field_jacobian(
        pred_phys, grid_x_raw, grid_y_raw, smooth_sigma=VORTEX_SMOOTH_SIGMA)

    return {
        "cp_gt_field": cp_gt_field, "cp_pred_field": cp_pred_field,
        "x_surface": x_surf, "y_surface": y_surf,
        "cp_gt_surface": cp_gt_surf, "cp_pred_surface": cp_pred_surf,
        "cd_gt": cd_gt, "cd_pred": cd_pred, "cl_gt": cl_gt, "cl_pred": cl_pred,
        "vort_gt": vort_gt, "vort_pred": vort_pred,
        "q_gt": q_gt, "q_pred": q_pred,
        "p_inf": p_inf, "rho_inf": rho_inf, "V_inf": V_inf, "q_inf": q_inf,
        "chord": chord_eff, "i_range": (i0, i1),
    }


#==========================================#5. 涡流误差指标（三种：绝对 / 相对 / 涡核）#==========================================def compute_vortex_metrics(vort_gt, vort_pred, metric_type="q_criterion",
                           core_percentile=75):
    eps = 1e-6
    diff_abs = np.abs(vort_gt - vort_pred)
    mae_abs = float(np.mean(diff_abs))

    #✅ 修正：全流场相对误差只在显著涡区计算    threshold = 0.01 * np.max(np.abs(vort_gt))
    significant_mask = np.abs(vort_gt) > threshold
    if np.any(significant_mask):
        mae_rel = float(np.mean(diff_abs[significant_mask] /
                                (np.abs(vort_gt[significant_mask]) + eps)))
    else:
        mae_rel = 0.0

    #涡核区域    if metric_type == "q_criterion":
        pos = vort_gt[vort_gt > 0]
        threshold_core = np.percentile(pos, core_percentile) if len(pos) > 0 else 0.0
        core_mask = vort_gt > threshold_core
    else:
        threshold_core = np.percentile(np.abs(vort_gt), core_percentile)
        core_mask = np.abs(vort_gt) > threshold_core

    if np.any(core_mask):
        core_mae = float(np.mean(diff_abs[core_mask]))
        core_rel = float(np.mean(diff_abs[core_mask] /
                                 (np.abs(vort_gt[core_mask]) + eps)))
    else:
        core_mae = 0.0
        core_rel = 0.0

    l2_err = float(np.linalg.norm(vort_gt - vort_pred) /
                   (np.linalg.norm(vort_gt) + eps) * 100)

    return {"MAE_abs": mae_abs, "MAE_rel": mae_rel,
            "Core_MAE": core_mae, "Core_REL": core_rel, "L2_err(%)": l2_err}

#==========================================#6. 数据集评估循环#==========================================def evaluate_selected_samples(dataset_name, data_npz, model, device,
                              grid_x_tensor, grid_y_tensor,
                              grid_x_raw, grid_y_raw,
                              f_min, f_max, l_min, l_max,
                              sample_indices, ddim_steps=50, batch_size=4):
    print(f"\n🚀 开始推理选定的 {len(sample_indices)} 个样本: 【{dataset_name}】")
    if len(sample_indices) == 0:
        print("⚠️ 样本索引为空。")
        return None, None, None, None

    x_data = data_npz['x']; y_data = data_npz['y']
    results = []; all_gt_phys = []; all_pred_phys = []; all_cp_cd = []

    for i in tqdm(range(0, len(sample_indices), batch_size), desc=f"Inferencing {dataset_name}"):
        batch_indices = sample_indices[i:i + batch_size]
        batch_x_norm = x_data[batch_indices]
        batch_y_norm = y_data[batch_indices]

        cond_tensor = torch.tensor(batch_y_norm, dtype=torch.float32, device=device)
        if cond_tensor.dim() == 1:
            cond_tensor = cond_tensor.unsqueeze(1)

        pred_norm_ddim = sample_ddim_batch(
            model, device, grid_x_tensor, grid_y_tensor, cond_tensor,
            total_timesteps=1000, ddim_steps=ddim_steps)

        gt_phys = denormalize_batch(batch_x_norm, f_min, f_max)
        pred_phys = denormalize_batch(pred_norm_ddim, f_min, f_max)

        all_gt_phys.append(gt_phys); all_pred_phys.append(pred_phys)

        for j, idx in enumerate(batch_indices):
            y_val = batch_y_norm[j].item() if hasattr(batch_y_norm[j], 'item') else float(batch_y_norm[j])
            time_val = float(y_val * (l_max - l_min + 1e-8) + l_min)

            cp_cd = compute_cp_cd_cl(gt_phys[j], pred_phys[j], grid_x_raw, grid_y_raw)
            all_cp_cd.append(cp_cd)

            cd_gt, cd_pred = cp_cd["cd_gt"], cp_cd["cd_pred"]
            cl_gt, cl_pred = cp_cd["cl_gt"], cp_cd["cl_pred"]
            cd_err = abs(cd_pred - cd_gt); cd_rel = 100.0 * cd_err / (abs(cd_gt) + 1e-10)
            cl_err = abs(cl_pred - cl_gt); cl_rel = 100.0 * cl_err / (abs(cl_gt) + 1e-10)

            vort_gt_sel = cp_cd["q_gt"] if VORTEX_METRIC == "q_criterion" else cp_cd["vort_gt"]
            vort_pr_sel = cp_cd["q_pred"] if VORTEX_METRIC == "q_criterion" else cp_cd["vort_pred"]
            vmet = compute_vortex_metrics(vort_gt_sel, vort_pr_sel,
                                          metric_type=VORTEX_METRIC,
                                          core_percentile=VORTEX_CORE_PERCENTILE)

            entry = {
                "sample_idx": idx, "time_step": time_val,
                "Cd_pressure_GT": cd_gt, "Cd_pressure_pred": cd_pred,
                "Cd_abs_err": cd_err, "Cd_rel_err(%)": cd_rel,
                "Cl_GT": cl_gt, "Cl_pred": cl_pred,
                "Cl_abs_err": cl_err, "Cl_rel_err(%)": cl_rel,
                "Vortex_MAE_abs": vmet["MAE_abs"],
                "Vortex_MAE_rel": vmet["MAE_rel"],
                "Vortex_Core_MAE": vmet["Core_MAE"],
                "Vortex_Core_REL": vmet["Core_REL"],
                "Vortex_L2_err(%)": vmet["L2_err(%)"],
            }
            results.append(entry)

    df = pd.DataFrame(results)
    all_gt_phys = np.concatenate(all_gt_phys, axis=0)
    all_pred_phys = np.concatenate(all_pred_phys, axis=0)
    print(f"✅ {dataset_name} 추론 완료!")
    return df, all_gt_phys, all_pred_phys, all_cp_cd


#==========================================#7. 涡流可视化（4行1列：GT/Pred/绝对误差/相对误差）#==========================================def _overlay_airfoil(ax, grid_x_raw, grid_y_raw, airfoil_idx=0,
                     color='k', lw=1.6, zorder=20, fill=False, facecolor='0.35'):
    x_s = grid_x_raw[airfoil_idx, :]; y_s = grid_y_raw[airfoil_idx, :]
    if fill:
        ax.fill(x_s, y_s, color=facecolor, alpha=0.85, zorder=zorder - 1)
    ax.plot(x_s, y_s, color=color, lw=lw, zorder=zorder)


def plot_vortex_comparison(grid_x_raw, grid_y_raw, gt_phys, pred_phys, cp_cd, time_val,
                           mode="q_criterion", airfoil_idx=0):
    u_gt, v_gt = gt_phys[1], gt_phys[2]
    u_pr, v_pr = pred_phys[1], pred_phys[2]

    if mode == "q_criterion":
        field_gt = cp_cd["q_gt"]; field_pr = cp_cd["q_pred"]
        title_label = "Q-Criterion (Q > 0)"; cmap = "viridis"
        pos_vals = field_gt[field_gt > 0]
        vmax = max(float(np.percentile(pos_vals, 90)) if len(pos_vals) > 0 else 1e-3, 1e-3)
        vmin = 0.0
    else:
        field_gt = cp_cd["vort_gt"]; field_pr = cp_cd["vort_pred"]
        title_label = "Vorticity ωz"; cmap = "coolwarm"
        vabs = max(float(np.percentile(np.abs(field_gt), 90)), 1e-3)
        vmin, vmax = -vabs, vabs

    field_diff_abs = np.abs(field_gt - field_pr)
    if mode == "q_criterion":
        field_diff_abs = np.clip(field_diff_abs, 0, vmax)

    #相对误差：加 eps 防除零；色标上限 2.0 (200%)    field_diff_rel = np.abs(field_gt - field_pr) / (np.abs(field_gt) + 1e-6)
    field_diff_rel = np.clip(field_diff_rel, 0, 2.0)

    skip = 8
    q_x = grid_x_raw[::skip, ::skip]; q_y = grid_y_raw[::skip, ::skip]
    q_u_gt = u_gt[::skip, ::skip]; q_v_gt = v_gt[::skip, ::skip]
    q_u_pr = u_pr[::skip, ::skip]; q_v_pr = v_pr[::skip, ::skip]

    #根据物理区域长宽比自适应画布    x_range = grid_x_raw.max() - grid_x_raw.min()
    y_range = grid_y_raw.max() - grid_y_raw.min()
    aspect_ratio = x_range / (y_range + 1e-8)
    fig_w = 14
    fig_h_per_row = np.clip(fig_w / aspect_ratio * 0.55, 1.5, 3.5)
    fig_h = fig_h_per_row * 4 + 1.5

    fig, axes = plt.subplots(4, 1, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.05, hspace=0.28, top=0.93, bottom=0.05,
                        left=0.07, right=0.91)

    #(1) GT    im0 = axes[0].pcolormesh(grid_x_raw, grid_y_raw, field_gt,
                             shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    axes[0].quiver(q_x, q_y, q_u_gt, q_v_gt, color='k',
                   scale=30, width=0.0012, alpha=0.75, zorder=5)
    _overlay_airfoil(axes[0], grid_x_raw, grid_y_raw, airfoil_idx, fill=True)
    axes[0].set_title(f"① Ground Truth  —  {title_label}",
                      fontsize=12, fontweight='bold', loc='left')
    axes[0].set_ylabel("Y (m)", fontsize=10); axes[0].set_aspect('equal')

    #(2) Prediction    im1 = axes[1].pcolormesh(grid_x_raw, grid_y_raw, field_pr,
                             shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    axes[1].quiver(q_x, q_y, q_u_pr, q_v_pr, color='k',
                   scale=30, width=0.0012, alpha=0.75, zorder=5)
    _overlay_airfoil(axes[1], grid_x_raw, grid_y_raw, airfoil_idx, fill=True)
    axes[1].set_title(f"② DDIM Prediction  —  {title_label}",
                      fontsize=12, fontweight='bold', loc='left')
    axes[1].set_ylabel("Y (m)", fontsize=10); axes[1].set_aspect('equal')

    #(3) 绝对误差    im2 = axes[2].pcolormesh(grid_x_raw, grid_y_raw, field_diff_abs,
                             shading='gouraud', cmap='magma_r',
                             vmin=0, vmax=vmax, rasterized=True)
    _overlay_airfoil(axes[2], grid_x_raw, grid_y_raw, airfoil_idx, color='cyan', lw=1.4)
    axes[2].set_title(
        f"③ 绝对误差 |Δ {mode}|  —  MAE = {np.mean(field_diff_abs):.4f}  (涡核主导)",
        fontsize=12, fontweight='bold', loc='left')
    axes[2].set_ylabel("Y (m)", fontsize=10); axes[2].set_aspect('equal')

    #(4) 相对误差    im3 = axes[3].pcolormesh(grid_x_raw, grid_y_raw, field_diff_rel,
                             shading='gouraud', cmap='magma_r',
                             vmin=0, vmax=2.0, rasterized=True)
    _overlay_airfoil(axes[3], grid_x_raw, grid_y_raw, airfoil_idx, color='cyan', lw=1.4)
    axes[3].set_title(
        f"④ 相对误差 |ΔQ|/|Q|  —  平均 = {np.mean(field_diff_rel):.3f}  (色标上限 200%)",
        fontsize=12, fontweight='bold', loc='left')
    axes[3].set_xlabel("X (m)", fontsize=10)
    axes[3].set_ylabel("Y (m)", fontsize=10); axes[3].set_aspect('equal')

    cbar_labels = [title_label, title_label, "Absolute Error", "Relative Error (clip 200%)"]
    for ax, im, label in zip(axes, [im0, im1, im2, im3], cbar_labels):
        cbar = fig.colorbar(im, ax=ax, location='right',
                            fraction=0.025, pad=0.012, aspect=25)
        cbar.set_label(label, fontsize=9, labelpad=5)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(f"涡流结构对比与速度矢量图  —  Time = {time_val:.1f}",
                 fontsize=14, fontweight='bold', y=0.975)
    plt.show(); plt.close(fig)


#==========================================#8. 主程序入口#==========================================if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🌟 {DATA_SUFFIX} 데이터셋 Diffusion 推理与 涡流/Cp/Cd/Cl 评估 | 设备: {device}")

    DDIM_STEPS = 50; BATCH_SIZE = 4

    current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
    results_dir = os.path.abspath(os.path.join(current_dir, "../Results"))

    weights_candidates = [
        os.path.join(results_dir, f"airfoil_diffusion_cgrid_{DATA_SUFFIX}_final{TARGET_LOSS}.pth"),
        os.path.join(results_dir, f"airfoil_diffusion_cgrid_{DATA_SUFFIX}_ep3000.pth"),
        os.path.join(results_dir, f"airfoil_diffusion_cgrid_{DATA_SUFFIX}_ep2500.pth"),
    ]
    weights_path = next((w for w in weights_candidates if os.path.exists(w)), None)
    if weights_path is None:
        raise FileNotFoundError(f"❌ 未找到权重文件 (后缀: {DATA_SUFFIX})")

    norm_path = os.path.join(results_dir, f"normalization_factors_{DATA_SUFFIX}.npz")
    test_data_path = os.path.join(results_dir, f"Diffusion_airfoil_unsteady_{DATA_SUFFIX}_test.npz")
    train_data_path = os.path.join(results_dir, f"Diffusion_airfoil_unsteady_{DATA_SUFFIX}_train.npz")

    model = DiffusionUNet(flow_ch=4, coord_ch=2, cond_dim=128, base_ch=48).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    print(f"✅ 模型权重加载成功: {weights_path}")

    train_data = np.load(train_data_path)
    test_data = np.load(test_data_path)
    norm_factors = np.load(norm_path)

    f_min, f_max = norm_factors["fields_min"], norm_factors["fields_max"]
    l_min = float(np.asarray(norm_factors["label_min"]).reshape(-1)[0])
    l_max = float(np.asarray(norm_factors["label_max"]).reshape(-1)[0])

    grid_x_raw = train_data["grid_x"]; grid_y_raw = train_data["grid_y"]
    grid_x_tensor = torch.tensor(grid_x_raw, dtype=torch.float32, device=device)
    grid_y_tensor = torch.tensor(grid_y_raw, dtype=torch.float32, device=device)
    grid_x_tensor = 2.0 * (grid_x_tensor - grid_x_tensor.min()) / (grid_x_tensor.max() - grid_x_tensor.min() + 1e-8) - 1.0
    grid_y_tensor = 2.0 * (grid_y_tensor - grid_y_tensor.min()) / (grid_y_tensor.max() - grid_y_tensor.min() + 1e-8) - 1.0
    if grid_x_tensor.dim() == 2:
        grid_x_tensor = grid_x_tensor.unsqueeze(0).unsqueeze(0)
        grid_y_tensor = grid_y_tensor.unsqueeze(0).unsqueeze(0)

    if EVAL_ALL or not TARGET_TIMESTEPS:
        print("🔍 【全量测试集】 와류/Cp/Cd/Cl 推理与评估中...")
        test_indices = list(range(len(test_data['x'])))
        df_test, test_gt, test_pred, test_cp_cd = evaluate_selected_samples(
            "Test Dataset", test_data, model, device,
            grid_x_tensor, grid_y_tensor, grid_x_raw, grid_y_raw,
            f_min, f_max, l_min, l_max,
            sample_indices=test_indices, ddim_steps=DDIM_STEPS, batch_size=BATCH_SIZE)

        if ENABLE_PLOTS:
            print(f"🎨 生成涡流 ({VORTEX_METRIC}) 与流线叠加图... (ENABLE_PLOTS=True)")
            best_sample_id = df_test["Cd_abs_err"].idxmin()
            best_relative_pos = df_test.index.get_loc(best_sample_id)
            row = df_test.iloc[best_relative_pos]
            plot_vortex_comparison(
                grid_x_raw, grid_y_raw,
                test_gt[best_relative_pos], test_pred[best_relative_pos],
                test_cp_cd[best_relative_pos],
                row['time_step'], mode=VORTEX_METRIC, airfoil_idx=AIRFOIL_SURFACE_INDEX)
        else:
            print("⚡ 已跳过图像生成 (ENABLE_PLOTS=False)")

        #==========================================================================        #综合报告（含涡流三种指标说明）        #==========================================================================        print("\n" + "=" * 125)
        print(f"📊 [{DATA_SUFFIX}] 测试集 Cp/Cd/Cl 及 와류({VORTEX_METRIC}) 评估总结 (样本数: {len(df_test)})")
        print("=" * 125)

        print("【涡流误差】")
        print(f"  ⚠️  绝对误差 MAE_abs    : {df_test['Vortex_MAE_abs'].mean():.6f}  (被涡核主导，仅作参考)")
        print(f"  ✅  相对误差 MAE_rel    : {df_test['Vortex_MAE_rel'].mean():.6f}  (全流场平均，推荐)")
        print(f"  ✅  涡核绝对误差 Core_MAE : {df_test['Vortex_Core_MAE'].mean():.6f}")
        print(f"  ✅  涡核相对误差 Core_REL : {df_test['Vortex_Core_REL'].mean():.6f}  (真实反映预测质量)")
        print(f"  ℹ️  相对 L2 오차          : {df_test['Vortex_L2_err(%)'].mean():.3f}%")
        print("  📝 注：와류 MAE 高是导数放大的固有特性，")
        print("         涡核相对误差 Core_REL 更能反映预测质量。")
        print("-" * 125)

        metrics_summary = [
            {"Metric": "Cd_pressure (压差阻力)",
             "MAE_Abs": df_test["Cd_abs_err"].mean(), "Max_Abs": df_test["Cd_abs_err"].max(),
             "Min_Abs": df_test["Cd_abs_err"].min(),
             "MAE_Rel(%)": df_test["Cd_rel_err(%)"].mean(),
             "Max_Rel(%)": df_test["Cd_rel_err(%)"].max(),
             "Min_Rel(%)": df_test["Cd_rel_err(%)"].min()},
            {"Metric": "Cl (升力系数)",
             "MAE_Abs": df_test["Cl_abs_err"].mean(), "Max_Abs": df_test["Cl_abs_err"].max(),
             "Min_Abs": df_test["Cl_abs_err"].min(),
             "MAE_Rel(%)": df_test["Cl_rel_err(%)"].mean(),
             "Max_Rel(%)": df_test["Cl_rel_err(%)"].max(),
             "Min_Rel(%)": df_test["Cl_rel_err(%)"].min()},
        ]
        df_metrics = pd.DataFrame(metrics_summary)

        if SAVE_METRICS_CSV:
            csv_path = os.path.join(results_dir, f"eval_{DATA_SUFFIX}_metrics_summary.csv")
            df_metrics.to_csv(csv_path, index=False)
            print(f"📁 指标汇总 CSV 저장: {csv_path}")

        print(f"{'物理气动指标':<22} | {'MAE(绝对)':<12} | {'Max(绝对)':<12} | {'Min(绝对)':<12} | "
              f"{'MAE(相对%)':<12} | {'Max(相对%)':<12} | {'Min(相对%)':<12}")
        print("-" * 125)
        for _, r in df_metrics.iterrows():
            print(f"{r['Metric']:<22} | {r['MAE_Abs']:<12.6f} | {r['Max_Abs']:<12.6f} | "
                  f"{r['Min_Abs']:<12.6f} | {r['MAE_Rel(%)']:<12.3f} | "
                  f"{r['Max_Rel(%)']:<12.3f} | {r['Min_Rel(%)']:<12.3f}")
        print("=" * 125 + "\n")