#!/usr/bin/python3#-*- coding: utf-8 -*-
"""
Auto size-adaptive (full / down4) C-grid DiffusionUNet inference and evaluation script.
Features: DDIM 50-step fast sampling + physical denormalization + Cp/Cd/Cl computation
          with visualization and CSV output.
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
from model_utils import DiffusionUNet

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.max_open_warning'] = 0

#==========================================#0. Configuration#==========================================DATA_SUFFIX = "down1"
TARGET_LOSS = "_pys"
TARGET_TIMESTEPS = [154]
EVAL_ALL = False

ENABLE_PLOTS = True
SAVE_METRICS_CSV = False

AIRFOIL_SURFACE_INDEX = 0
AIRFOIL_CHORD_GUESS   = 1.0
AIRFOIL_Y_TOL         = 0.30
CHORD_LENGTH_OVERRIDE = None

#==========================================#1. DDIM deterministic fast sampler#==========================================@torch.no_grad()
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

#==========================================#2. Physical denormalization#==========================================def denormalize_batch(field_tensor, f_min, f_max):
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

#==========================================#3. Cp / Cd / Cl integration#==========================================def estimate_freestream_conditions(gt_phys):
    p   = gt_phys[3]
    u   = gt_phys[1]
    v   = gt_phys[2]
    rho = gt_phys[0]

    edge_mask = np.zeros_like(p, dtype=bool)
    edge_mask[-1, :]  = True
    edge_mask[1:, 0]  = True
    edge_mask[1:, -1] = True

    p_inf   = float(np.mean(p[edge_mask]))
    u_inf   = float(np.mean(u[edge_mask]))
    v_inf   = float(np.mean(v[edge_mask]))
    rho_inf = float(np.mean(rho[edge_mask]))
    V_inf   = float(np.sqrt(u_inf**2 + v_inf**2))

    return p_inf, rho_inf, V_inf

def compute_cp_field(pressure, p_ref, q_ref):
    return (pressure - p_ref) / (q_ref + 1e-10)

def compute_cd_pressure_green(x_surf, y_surf, cp_surf, chord):
    x  = np.append(x_surf, x_surf[0])
    y  = np.append(y_surf, y_surf[0])
    cp = np.append(cp_surf, cp_surf[0])

    signed_area_2 = np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    is_ccw = signed_area_2 > 0

    dy     = np.diff(y)
    cp_mid = 0.5 * (cp[:-1] + cp[1:])
    integral_cp_dy = np.sum(cp_mid * dy)

    cd = (-integral_cp_dy if is_ccw else integral_cp_dy) / (chord + 1e-10)
    return float(cd)

def compute_cl_pressure_green(x_surf, y_surf, cp_surf, chord):
    x  = np.append(x_surf, x_surf[0])
    y  = np.append(y_surf, y_surf[0])
    cp = np.append(cp_surf, cp_surf[0])

    signed_area_2 = np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    is_ccw = signed_area_2 > 0

    dx     = np.diff(x)
    cp_mid = 0.5 * (cp[:-1] + cp[1:])
    integral_cp_dx = np.sum(cp_mid * dx)

    cl = (integral_cp_dx if is_ccw else -integral_cp_dx) / (chord + 1e-10)
    return float(cl)

def find_airfoil_segment(grid_x_raw, grid_y_raw, airfoil_idx=0,
                         chord_guess=1.0, y_tol=0.30, verbose=False):
    x = grid_x_raw[airfoil_idx, :]
    y = grid_y_raw[airfoil_idx, :]

    x_lo = -0.05 * chord_guess
    x_hi =  0.975 * chord_guess
    mask = (x >= x_lo) & (x <= x_hi) & (np.abs(y) < y_tol * chord_guess)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise RuntimeError(f"Could not find airfoil segment in row {airfoil_idx}.")

    breaks = np.where(np.diff(idx) > 1)[0]
    if len(breaks) == 0:
        i0, i1 = int(idx[0]), int(idx[-1]) + 1
    else:
        segs = np.split(idx, breaks + 1)
        longest = max(segs, key=len)
        i0, i1 = int(longest[0]), int(longest[-1]) + 1

    if verbose:
        print(f"[Airfoil segment] row={airfoil_idx} i in [{i0},{i1}) total {i1-i0} nodes")

    return i0, i1

def compute_cp_cd_cl(gt_phys, pred_phys, grid_x_raw, grid_y_raw,
                     airfoil_idx=AIRFOIL_SURFACE_INDEX,
                     airfoil_i_range=None,
                     chord_guess=AIRFOIL_CHORD_GUESS,
                     y_tol=AIRFOIL_Y_TOL,
                     chord_override=CHORD_LENGTH_OVERRIDE):

    p_inf, rho_inf, V_inf = estimate_freestream_conditions(gt_phys)
    q_inf = 0.5 * rho_inf * V_inf**2

    cp_gt_field   = compute_cp_field(gt_phys[3],   p_inf, q_inf)
    cp_pred_field = compute_cp_field(pred_phys[3], p_inf, q_inf)

    if airfoil_i_range is None:
        i0, i1 = find_airfoil_segment(grid_x_raw, grid_y_raw,
                                      airfoil_idx=airfoil_idx,
                                      chord_guess=chord_guess,
                                      y_tol=y_tol, verbose=False)
    else:
        i0, i1 = airfoil_i_range

    x_surf = grid_x_raw[airfoil_idx, i0:i1]
    y_surf = grid_y_raw[airfoil_idx, i0:i1]
    cp_gt_surf   = cp_gt_field[airfoil_idx, i0:i1]
    cp_pred_surf = cp_pred_field[airfoil_idx, i0:i1]

    chord_eff = float(np.max(x_surf) - np.min(x_surf)) if chord_override is None else float(chord_override)

    cd_gt   = compute_cd_pressure_green(x_surf, y_surf, cp_gt_surf,   chord_eff)
    cd_pred = compute_cd_pressure_green(x_surf, y_surf, cp_pred_surf, chord_eff)

    cl_gt   = compute_cl_pressure_green(x_surf, y_surf, cp_gt_surf,   chord_eff)
    cl_pred = compute_cl_pressure_green(x_surf, y_surf, cp_pred_surf, chord_eff)

    return {
        "cp_gt_field":     cp_gt_field,
        "cp_pred_field":   cp_pred_field,
        "x_surface":       x_surf,
        "y_surface":       y_surf,
        "cp_gt_surface":   cp_gt_surf,
        "cp_pred_surface": cp_pred_surf,
        "cd_gt":           cd_gt,
        "cd_pred":         cd_pred,
        "cl_gt":           cl_gt,
        "cl_pred":         cl_pred,
        "p_inf":           p_inf,
        "rho_inf":         rho_inf,
        "V_inf":           V_inf,
        "q_inf":           q_inf,
        "chord":           chord_eff,
        "i_range":         (i0, i1),
    }

#==========================================#4. Dataset evaluation / inference loop#==========================================def evaluate_selected_samples(dataset_name, data_npz, model, device,
                              grid_x_tensor, grid_y_tensor,
                              grid_x_raw, grid_y_raw,
                              f_min, f_max, l_min, l_max,
                              sample_indices, ddim_steps=50, batch_size=4):
    print(f"\nStarting inference on {len(sample_indices)} selected samples: [{dataset_name}]")

    if len(sample_indices) == 0:
        print("Sample index list is empty.")
        return None, None, None, None

    x_data = data_npz['x']
    y_data = data_npz['y']
    results = []
    all_gt_phys = []
    all_pred_phys = []
    all_cp_cd = []

    for i in tqdm(range(0, len(sample_indices), batch_size), desc=f"Inferencing {dataset_name}"):
        batch_indices = sample_indices[i:i + batch_size]
        batch_x_norm = x_data[batch_indices]
        batch_y_norm = y_data[batch_indices]

        cond_tensor = torch.tensor(batch_y_norm, dtype=torch.float32, device=device)
        if cond_tensor.dim() == 1:
            cond_tensor = cond_tensor.unsqueeze(1)

        pred_norm_ddim = sample_ddim_batch(
            model, device, grid_x_tensor, grid_y_tensor, cond_tensor,
            total_timesteps=1000, ddim_steps=ddim_steps
        )

        gt_phys   = denormalize_batch(batch_x_norm, f_min, f_max)
        pred_phys = denormalize_batch(pred_norm_ddim, f_min, f_max)

        all_gt_phys.append(gt_phys)
        all_pred_phys.append(pred_phys)

        for j, idx in enumerate(batch_indices):
            y_val = float(np.asarray(batch_y_norm[j]).reshape(-1)[0])
            time_val = float(y_val * (l_max - l_min + 1e-8) + l_min)

            cp_cd = compute_cp_cd_cl(gt_phys[j], pred_phys[j], grid_x_raw, grid_y_raw)
            all_cp_cd.append(cp_cd)

            cd_gt, cd_pred = cp_cd["cd_gt"], cp_cd["cd_pred"]
            cl_gt, cl_pred = cp_cd["cl_gt"], cp_cd["cl_pred"]

            cd_err = abs(cd_pred - cd_gt)
            cd_rel = 100.0 * cd_err / (abs(cd_gt) + 1e-10)

            cl_err = abs(cl_pred - cl_gt)
            cl_rel = 100.0 * cl_err / (abs(cl_gt) + 1e-10)

            entry = {
                "sample_idx":    idx,
                "time_step":     time_val,
                "Cd_GT":         cd_gt,
                "Cd_pred":       cd_pred,
                "Cd_abs_err":    cd_err,
                "Cd_rel_err(%)": cd_rel,
                "Cl_GT":         cl_gt,
                "Cl_pred":       cl_pred,
                "Cl_abs_err":    cl_err,
                "Cl_rel_err(%)": cl_rel,
                "Cp_min_GT":     float(np.min(cp_cd["cp_gt_surface"])),
                "Cp_min_pred":   float(np.min(cp_cd["cp_pred_surface"])),
                "Cp_mean_GT":    float(np.mean(cp_cd["cp_gt_surface"])),
                "Cp_mean_pred":  float(np.mean(cp_cd["cp_pred_surface"])),
                "p_inf":         cp_cd["p_inf"],
                "V_inf":         cp_cd["V_inf"],
                "chord":         cp_cd["chord"],
            }
            results.append(entry)

    df = pd.DataFrame(results)
    all_gt_phys   = np.concatenate(all_gt_phys, axis=0)
    all_pred_phys = np.concatenate(all_pred_phys, axis=0)

    print(f"{dataset_name} selected-sample inference complete.")
    return df, all_gt_phys, all_pred_phys, all_cp_cd

#==========================================#5. Visualization helpers#==========================================def _overlay_airfoil(ax, grid_x_raw, grid_y_raw, airfoil_idx=0,
                     color='k', lw=1.6, zorder=20, fill=False, facecolor='0.35'):
    x_s = grid_x_raw[airfoil_idx, :]
    y_s = grid_y_raw[airfoil_idx, :]
    if fill:
        ax.fill(x_s, y_s, color=facecolor, alpha=0.85, zorder=zorder - 1)
    ax.plot(x_s, y_s, color=color, lw=lw, zorder=zorder)

def _split_upper_lower(x_surf, y_surf, cp_gt_surf, cp_pred_surf):
    le_idx = int(np.argmin(x_surf))

    seg_a_x  = x_surf[:le_idx + 1]
    seg_a_y  = y_surf[:le_idx + 1]
    seg_a_gt = cp_gt_surf[:le_idx + 1]
    seg_a_pr = cp_pred_surf[:le_idx + 1]

    seg_b_x  = x_surf[le_idx:]
    seg_b_y  = y_surf[le_idx:]
    seg_b_gt = cp_gt_surf[le_idx:]
    seg_b_pr = cp_pred_surf[le_idx:]

    mean_y_a = float(np.mean(seg_a_y))
    mean_y_b = float(np.mean(seg_b_y))

    if mean_y_a >= mean_y_b:
        upper = (seg_a_x, seg_a_gt, seg_a_pr)
        lower = (seg_b_x, seg_b_gt, seg_b_pr)
    else:
        upper = (seg_b_x, seg_b_gt, seg_b_pr)
        lower = (seg_a_x, seg_a_gt, seg_a_pr)
    return upper, lower

def plot_cp_comparison(grid_x_raw, grid_y_raw, cp_cd, time_val,
                       airfoil_idx=0, sample_label=None):
    cp_gt   = cp_cd["cp_gt_field"]
    cp_pred = cp_cd["cp_pred_field"]
    cp_diff = np.abs(cp_gt - cp_pred)

    vmin = float(min(cp_gt.min(), cp_pred.min()))
    vmax = float(max(cp_gt.max(), cp_pred.max()))
    vabs = max(abs(vmin), abs(vmax))
    vmin_sym, vmax_sym = (-vabs, vabs) if (vmin < 0 and vmax > 0) else (vmin, vmax)

    diff_max = float(cp_diff.max()) + 1e-10

    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1.35, 1.0],
        width_ratios=[1, 1, 1],
        hspace=0.30, wspace=0.22,
        left=0.05, right=0.965, top=0.90, bottom=0.07,
    )

    cmap_cp  = 'coolwarm'
    cmap_err = 'magma_r'

    #(0,0) Ground Truth Cp    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.pcolormesh(grid_x_raw, grid_y_raw, cp_gt, shading='gouraud',
                         cmap=cmap_cp, vmin=vmin_sym, vmax=vmax_sym, rasterized=True)
    _overlay_airfoil(ax1, grid_x_raw, grid_y_raw, airfoil_idx,
                     color='k', lw=1.6, fill=True, facecolor='0.25')
    ax1.set_title("(1) Ground Truth  Cp", fontsize=12, fontweight='bold', loc='left')
    ax1.set_aspect('equal')
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)")

    #(0,1) Diffusion prediction Cp    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.pcolormesh(grid_x_raw, grid_y_raw, cp_pred, shading='gouraud',
                         cmap=cmap_cp, vmin=vmin_sym, vmax=vmax_sym, rasterized=True)
    _overlay_airfoil(ax2, grid_x_raw, grid_y_raw, airfoil_idx,
                     color='k', lw=1.6, fill=True, facecolor='0.25')
    ax2.set_title("(2) Diffusion Prediction  Cp", fontsize=12, fontweight='bold', loc='left')
    ax2.set_aspect('equal')
    ax2.set_xlabel("X (m)")

    cbar_ax = fig.add_axes([0.335, 0.945, 0.31, 0.018])
    cbar = fig.colorbar(im2, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Cp  ( (p - p_inf) / q_inf )", fontsize=10)

    #(0,2) Difference    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.pcolormesh(grid_x_raw, grid_y_raw, cp_diff, shading='gouraud',
                         cmap=cmap_err, vmin=0.0, vmax=diff_max, rasterized=True)
    _overlay_airfoil(ax3, grid_x_raw, grid_y_raw, airfoil_idx,
                     color='cyan', lw=1.6, fill=False)
    ax3.set_title(f"(3) |dCp| field   Max = {diff_max:.3f}",
                  fontsize=12, fontweight='bold', loc='left')
    ax3.set_aspect('equal')
    ax3.set_xlabel("X (m)")
    cb3 = fig.colorbar(im3, ax=ax3, fraction=0.045, pad=0.02)
    cb3.set_label("|dCp|", fontsize=9)

    #(1,0)-(1,1) Surface Cp    ax4 = fig.add_subplot(gs[1, :2])
    x_surf       = cp_cd["x_surface"]
    y_surf       = cp_cd["y_surface"]
    cp_gt_surf   = cp_cd["cp_gt_surface"]
    cp_pred_surf = cp_cd["cp_pred_surface"]
    chord        = cp_cd["chord"]

    x_min = x_surf.min()
    def _xc(x): return (x - x_min) / (chord + 1e-10)

    upper, lower = _split_upper_lower(x_surf, y_surf, cp_gt_surf, cp_pred_surf)
    x_up, gt_up, pr_up = upper
    x_lo, gt_lo, pr_lo = lower
    xc_up, xc_lo = _xc(x_up), _xc(x_lo)

    C_UP_GT, C_UP_PR = '#1f4e9c', '#3aa1ff'
    C_LO_GT, C_LO_PR = '#8b1a1a', '#ff5a5a'

    ax4.plot(xc_up, gt_up, color=C_UP_GT, lw=2.4, ls='-',  label='Upper GT',  zorder=6)
    ax4.plot(xc_lo, gt_lo, color=C_LO_GT, lw=2.4, ls='-',  label='Lower GT',  zorder=6)
    ax4.plot(xc_up, pr_up, color=C_UP_PR, lw=2.0, ls='--', label='Upper Pred', zorder=7)
    ax4.plot(xc_lo, pr_lo, color=C_LO_PR, lw=2.0, ls='--', label='Lower Pred', zorder=7)

    ax4.invert_yaxis()
    ax4.set_xlabel("x / c", fontsize=11)
    ax4.set_ylabel("Cp", fontsize=11)

    cp_min_gt_val = cp_gt_surf.min()
    cp_min_pr_val = cp_pred_surf.min()
    ax4.set_title(
        f"(4) Airfoil Surface Cp Distribution  |  "
        f"MAE(Cp_surf) = {np.mean(np.abs(cp_gt_surf - cp_pred_surf)):.4f}  |  "
        f"Cp_min: GT={cp_min_gt_val:.3f}, Pred={cp_min_pr_val:.3f}",
        fontsize=12, fontweight='bold', loc='left'
    )
    ax4.grid(True, alpha=0.3, ls='--')
    ax4.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)

        #(1,2) Cd comparison    ax5 = fig.add_subplot(gs[1, 2])
    cd_gt   = cp_cd["cd_gt"]
    cd_pred = cp_cd["cd_pred"]
    cd_err  = abs(cd_pred - cd_gt)
    cd_rel  = 100.0 * cd_err / (abs(cd_gt) + 1e-10)

    bars = ax5.bar(['GT', 'Predicted'], [cd_gt, cd_pred],
                   color=['#2c3e50', '#e74c3c'], width=0.55,
                   edgecolor='black', linewidth=1.0, zorder=3)

    ymax_val = max(abs(cd_gt), abs(cd_pred)) if max(abs(cd_gt), abs(cd_pred)) > 0 else 1e-3
    ax5.set_ylim(0, ymax_val * 1.35)

    for bar, val in zip(bars, [cd_gt, cd_pred]):
        offset = 0.02 * ymax_val
        ax5.text(bar.get_x() + bar.get_width() / 2, val + offset, f'{val:.5f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax5.axhline(0, color='k', lw=0.8, alpha=0.5)

    ax5.text(0.5, 0.05,
             f"|dCd| = {cd_err:.5f}\nRel. Err = {cd_rel:.2f}%",
             transform=ax5.transAxes, ha='center', va='bottom', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff7bc',
                       edgecolor='#d4a017', lw=1.0))

    ax5.set_ylabel("Cd  (pressure)", fontsize=11)
    ax5.set_title("(5) Cd Comparison", fontsize=11, fontweight='bold', loc='left')
    ax5.grid(True, axis='y', alpha=0.3, ls='--', zorder=0)

    title = f"Airfoil Cp / Cd Prediction Comparison  -  Time = {time_val:.1f}"
    if sample_label is not None:
        title += f"   ({sample_label})"
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.985)

    plt.show()

#==========================================#6. Main entry#==========================================if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting diffusion inference and Cp/Cd/Cl evaluation | device: {device}")

    DDIM_STEPS = 50
    BATCH_SIZE = 4

    current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
    results_dir = os.path.abspath(os.path.join(current_dir, "../Results"))

    weights_candidates = [
        os.path.join(results_dir, f"airfoil_diffusion_cgrid_{DATA_SUFFIX}_final{TARGET_LOSS}.pth"),
        os.path.join(results_dir, f"airfoil_diffusion_cgrid_{DATA_SUFFIX}_ep3000.pth"),
        os.path.join(results_dir, f"airfoil_diffusion_cgrid_{DATA_SUFFIX}_ep2500.pth"),
    ]
    weights_path = next((w for w in weights_candidates if os.path.exists(w)), None)
    if weights_path is None:
        raise FileNotFoundError(f"Weight file not found (suffix: {DATA_SUFFIX})")

    norm_path = os.path.join(results_dir, f"normalization_factors_{DATA_SUFFIX}.npz")
    test_data_path  = os.path.join(results_dir, f"Diffusion_airfoil_unsteady_{DATA_SUFFIX}_test.npz")
    train_data_path = os.path.join(results_dir, f"Diffusion_airfoil_unsteady_{DATA_SUFFIX}_train.npz")

    model = DiffusionUNet(flow_ch=4, coord_ch=2, cond_dim=128, base_ch=48).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    print(f"Model weights loaded: {weights_path}")

    train_data   = np.load(train_data_path)
    test_data    = np.load(test_data_path)
    norm_factors = np.load(norm_path)

    f_min, f_max = norm_factors["fields_min"], norm_factors["fields_max"]
    l_min, l_max = norm_factors["label_min"],  norm_factors["label_max"]

    l_min = float(np.asarray(l_min).reshape(-1)[0])
    l_max = float(np.asarray(l_max).reshape(-1)[0])

    grid_x_raw = train_data["grid_x"]
    grid_y_raw = train_data["grid_y"]

    grid_x_tensor = torch.tensor(grid_x_raw, dtype=torch.float32, device=device)
    grid_y_tensor = torch.tensor(grid_y_raw, dtype=torch.float32, device=device)
    grid_x_tensor = 2.0 * (grid_x_tensor - grid_x_tensor.min()) / \
                    (grid_x_tensor.max() - grid_x_tensor.min() + 1e-8) - 1.0
    grid_y_tensor = 2.0 * (grid_y_tensor - grid_y_tensor.min()) / \
                    (grid_y_tensor.max() - grid_y_tensor.min() + 1e-8) - 1.0
    if grid_x_tensor.dim() == 2:
        grid_x_tensor = grid_x_tensor.unsqueeze(0).unsqueeze(0)
        grid_y_tensor = grid_y_tensor.unsqueeze(0).unsqueeze(0)

        #==========================================    #Build list of samples to evaluate    #==========================================    if EVAL_ALL:
        test_indices = list(range(len(test_data['x'])))
    else:
        #反解 time_step        y_all = np.asarray(test_data['y']).reshape(len(test_data['y']), -1)
        y_scalar = y_all[:, 0] if y_all.shape[1] == 1 else y_all.mean(axis=1)
        time_all = y_scalar * (l_max - l_min + 1e-8) + l_min

        print("Available time steps:", np.round(time_all, 3))

        test_indices = []
        for t_target in TARGET_TIMESTEPS:
            closest_idx = int(np.argmin(np.abs(time_all - t_target)))
            print(f"Target t = {t_target}, closest idx = {closest_idx}, "
                  f"actual t = {time_all[closest_idx]:.3f}")
            test_indices.append(closest_idx)

    if len(test_indices) == 0:
        print("No samples to evaluate. Exiting.")
        exit(0)

    #==========================================    #Run evaluation    #==========================================    print(f"Running Cp / Cd / Cl evaluation on {len(test_indices)} sample(s) ...")

    df_test, test_gt, test_pred, test_cp_cd = evaluate_selected_samples(
        "Test Dataset", test_data, model, device,
        grid_x_tensor, grid_y_tensor, grid_x_raw, grid_y_raw,
        f_min, f_max, l_min, l_max,
        sample_indices=test_indices,
        ddim_steps=DDIM_STEPS, batch_size=BATCH_SIZE
    )

    test_csv = os.path.join(results_dir, f"eval_{DATA_SUFFIX}_test_cp_cd.csv")
    df_test.to_csv(test_csv, index=False)
    print(f"Evaluation CSV saved: {test_csv}")

    #==========================================    #Visualization    #==========================================    if ENABLE_PLOTS:
        print("Plot visualization enabled. (ENABLE_PLOTS = True)")
        for k, orig_pos in enumerate(test_indices):
            row = df_test.iloc[k]
            plot_cp_comparison(
                grid_x_raw, grid_y_raw,
                test_cp_cd[k],
                row['time_step'],
                airfoil_idx=AIRFOIL_SURFACE_INDEX,
                sample_label=f"frame #{int(row['sample_idx'])}",
            )
    else:
        print("Skipping visualization. (ENABLE_PLOTS = False)")

    #==========================================    #Metrics summary    #==========================================    metrics_summary = [
        {
            "Metric": "Cd",
            "MAE_Abs": df_test["Cd_abs_err"].mean(),
            "Max_Abs": df_test["Cd_abs_err"].max(),
            "Min_Abs": df_test["Cd_abs_err"].min(),
            "MAE_Rel(%)": df_test["Cd_rel_err(%)"].mean(),
            "Max_Rel(%)": df_test["Cd_rel_err(%)"].max(),
            "Min_Rel(%)": df_test["Cd_rel_err(%)"].min(),
        },
        {
            "Metric": "Cl",
            "MAE_Abs": df_test["Cl_abs_err"].mean(),
            "Max_Abs": df_test["Cl_abs_err"].max(),
            "Min_Abs": df_test["Cl_abs_err"].min(),
            "MAE_Rel(%)": df_test["Cl_rel_err(%)"].mean(),
            "Max_Rel(%)": df_test["Cl_rel_err(%)"].max(),
            "Min_Rel(%)": df_test["Cl_rel_err(%)"].min(),
        }
    ]
    df_metrics = pd.DataFrame(metrics_summary)

    if SAVE_METRICS_CSV:
        summary_csv_path = os.path.join(results_dir, f"eval_{DATA_SUFFIX}_metrics_summary.csv")
        df_metrics.to_csv(summary_csv_path, index=False)
        print(f"Metrics summary CSV saved: {summary_csv_path}")

    print("\n" + "="*110)
    print(f"Cd and Cl prediction accuracy summary [{DATA_SUFFIX}] (total samples: {len(df_test)})")
    print("="*110)
    print(f"{'Metric':<10} | {'MAE Abs':<12} | {'Max Abs':<12} | {'Min Abs':<12} | "
          f"{'MAE Rel(%)':<12} | {'Max Rel(%)':<12} | {'Min Rel(%)':<12}")
    print("-" * 110)
    for _, r in df_metrics.iterrows():
        print(f"{r['Metric']:<10} | {r['MAE_Abs']:<12.6f} | {r['Max_Abs']:<12.6f} | "
              f"{r['Min_Abs']:<12.6f} | {r['MAE_Rel(%)']:<12.3f} | "
              f"{r['Max_Rel(%)']:<12.3f} | {r['Min_Rel(%)']:<12.3f}")
    print("="*110 + "\n")