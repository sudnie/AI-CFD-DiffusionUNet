#/usr/bin/python3
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# 🌟 1. 引入高保真、自适应长方形架构网络
from model_utils import HighResDiffusionUNet

# ==========================================
# 1. 高保真方案 B 反向降噪演进采样器 (60x301)
# ==========================================
@torch.no_grad()
def sample_flow_field_scheme_b(model, device, grid_x, grid_y, target_cond, timesteps=1000):
    model.eval()
    beta = torch.linspace(1e-4, 0.02, timesteps).to(device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    
    # 🎯 动态自适应大网格尺寸，直接在 (60, 301) 原生大空间里注入高斯噪声开始反向演进
    h, w = grid_x.shape[1], grid_x.shape[2]
    x_t = torch.randn((1, 4, h, w), device=device)
    
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
            x_t = mean + torch.sqrt(var) * torch.randn_like(x_t)
        else:
            x_t = x0_pred
    return x_t


# ==========================================
# 2. 来流条件估计（p_inf, rho_inf, V_inf, q_inf）
# ==========================================
def estimate_freestream_conditions(p_field, rho_field, u_field, v_field):
    """
    从 C-grid 外边界提取来流参考条件。
    外边界：最后一行 + 第一列 + 最后一列。
    """
    edge_mask = np.zeros_like(p_field, dtype=bool)
    edge_mask[-1, :]  = True
    edge_mask[1:, 0]  = True
    edge_mask[1:, -1] = True

    p_inf   = float(np.mean(p_field[edge_mask]))
    rho_inf = float(np.mean(rho_field[edge_mask]))
    u_inf   = float(np.mean(u_field[edge_mask]))
    v_inf   = float(np.mean(v_field[edge_mask]))
    V_inf   = float(np.sqrt(u_inf ** 2 + v_inf ** 2))
    q_inf   = 0.5 * rho_inf * V_inf ** 2

    return p_inf, rho_inf, V_inf, q_inf


# ==========================================
# 3. 原生高分辨率壁面 Cp 多层贴体平均提取与气动力算子
# ==========================================
def extract_raw_cp_and_forces(grid_x, grid_y, physical_press,
                              rho_field, u_field, v_field,
                              num_layers=5):
    """
    从物理压力场提取表面 Cp 并积分 Cl / Cd。

    流程:
        1. 从外边界估计来流动压 q_inf
        2. 物理压力 p → 压力系数 Cp = (p - p_inf) / q_inf
        3. 贴体多层平均提取表面 Cp
        4. 沿弦向排序、上下表面分割
        5. 积分 Cl / Cd
    """
    # ---------- 1. 估算来流动压 ----------
    p_inf, rho_inf, V_inf, q_inf = estimate_freestream_conditions(
        physical_press, rho_field, u_field, v_field
    )

    # ---------- 2. 物理压力 → Cp ----------
    cp_field = (physical_press - p_inf) / (q_inf + 1e-10)

    # ---------- 3. 贴体多层平均 ----------
    x_wall  = np.mean(grid_x[:num_layers, :], axis=0)
    y_wall  = np.mean(grid_y[:num_layers, :], axis=0)
    cp_wall = np.mean(cp_field[:num_layers, :], axis=0)

    # ---------- 4. 弦向过滤 [0, 1] ----------
    chord_mask = (x_wall >= 0.0) & (x_wall <= 1.0)
    x_f  = x_wall[chord_mask]
    y_f  = y_wall[chord_mask]
    cp_f = cp_wall[chord_mask]

    # ---------- 5. 上下表面分割 ----------
    upper_mask = y_f >= 0
    lower_mask = y_f < 0

    sort_up = np.argsort(x_f[upper_mask])
    sort_lo = np.argsort(x_f[lower_mask])

    x_up = x_f[upper_mask][sort_up]
    y_up = y_f[upper_mask][sort_up]
    cp_up = cp_f[upper_mask][sort_up]

    x_lo = x_f[lower_mask][sort_lo]
    y_lo = y_f[lower_mask][sort_lo]
    cp_lo = cp_f[lower_mask][sort_lo]

    # ---------- 5b. 前缘闭合处理 ----------
    # 上表面
    if x_up[0] > 0:
        slope_up = (cp_up[1] - cp_up[0]) / (x_up[1] - x_up[0] + 1e-10)
        cp_up_le = cp_up[0] - slope_up * x_up[0]

        x_up  = np.concatenate([[0.0], x_up])
        cp_up = np.concatenate([[cp_up_le], cp_up])
        y_up  = np.concatenate([[0.0], y_up])   # ← 补 y

    # 下表面
    if x_lo[0] > 0:
        slope_lo = (cp_lo[1] - cp_lo[0]) / (x_lo[1] - x_lo[0] + 1e-10)
        cp_lo_le = cp_lo[0] - slope_lo * x_lo[0]

        x_lo  = np.concatenate([[0.0], x_lo])
        cp_lo = np.concatenate([[cp_lo_le], cp_lo])
        y_lo  = np.concatenate([[0.0], y_lo])   # ← 补 y

    # 强制让上下表面在前缘处取同一个 Cp 值
    cp_le = 0.5 * (cp_up[0] + cp_lo[0])
    cp_up[0] = cp_le
    cp_lo[0] = cp_le
    # ---------- 6. Cl 积分 ----------
    cl_upper = np.trapz(cp_up, x_up)
    cl_lower = np.trapz(cp_lo, x_lo)
    cl = cl_lower - cl_upper

    # ---------- 7. Cd 积分（压差阻力）----------
    cdp_upper = np.sum(0.5 * (cp_up[:-1] + cp_up[1:]) * (y_up[1:] - y_up[:-1]))
    cdp_lower = np.sum(0.5 * (cp_lo[:-1] + cp_lo[1:]) * (y_lo[1:] - y_lo[:-1]))
    cdp = abs(cdp_upper - cdp_lower)

    return x_up, cp_up, x_lo, cp_lo, cl, cdp, p_inf, q_inf


# ==========================================
# 4. 自动化任务分配与流程控制引擎
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动无插值高分辨率流场特征提取与气动力评估引擎: {device}")

    # -------------------------------------------------------------
    # ⚙️ 核心流程控制器
    # -------------------------------------------------------------
    EVAL_MODE = "single"   # "single": 单工况验证 | "batch": 4工况竖排长图

    SINGLE_MACH = 0.475
    SINGLE_AOA  = 6

    BATCH_CASES = [
        {"Ma": 0.325, "AoA": 1.0}, {"Ma": 0.375, "AoA": 3.5},
        {"Ma": 0.425, "AoA": 6.0}, {"Ma": 0.475, "AoA": 6.0}
    ]

    if EVAL_MODE == "single":
        current_tasks = [{"Ma": SINGLE_MACH, "AoA": SINGLE_AOA}]
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]
        print(f"⚡ [单图快速验证模式] 开启：仅锁定 Mach={SINGLE_MACH}, AoA={SINGLE_AOA}°...")
    else:
        current_tasks = BATCH_CASES
        fig, axes = plt.subplots(4, 1, figsize=(8, 18), sharex=True)
        axes = axes.flatten()
        print(f"🌪️ [全量批量清算模式] 开启：依次计算全套 4 大工况阵列并合并为竖排长图...")

    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else "."
    weights_path    = os.path.abspath(os.path.join(current_dir, "../Results/airfoil_diffusion_mse_mae_ep2000.pth"))
    norm_path       = os.path.abspath(os.path.join(current_dir, "../Results/normalization_factors_highres.npz"))
    test_data_path  = os.path.abspath(os.path.join(current_dir, "../Results/HighRes_airfoil_test.npz"))
    train_data_path = os.path.abspath(os.path.join(current_dir, "../Results/HighRes_airfoil_train.npz"))

    model = HighResDiffusionUNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))

    test_data    = np.load(test_data_path)
    norm_factors = np.load(norm_path)

    f_min, f_max = norm_factors['fields_min'], norm_factors['fields_max']
    l_min, l_max = norm_factors['label_min'],  norm_factors['label_max']

    grid_x_high = test_data['grid_x']
    grid_y_high = test_data['grid_y']

    # -------------------------------------------------------------
    # 反归一化工具函数
    # -------------------------------------------------------------
    def denormalize_channel(field_norm_data, ch):
        """[4,H,W] → [H,W] 物理量，指定 channel"""
        arr = field_norm_data.cpu().numpy() if isinstance(field_norm_data, torch.Tensor) else field_norm_data
        if len(arr.shape) == 4:
            arr = arr[0]
        phys_0_1 = (arr + 1.0) / 2.0
        return phys_0_1[ch] * (f_max[0, ch, 0, 0] - f_min[0, ch, 0, 0]) + f_min[0, ch, 0, 0]

    def denormalize_all(field_norm_data):
        """返回 rho, u, v, p 四个物理场"""
        return {
            'rho': denormalize_channel(field_norm_data, 0),
            'u':   denormalize_channel(field_norm_data, 1),
            'v':   denormalize_channel(field_norm_data, 2),
            'p':   denormalize_channel(field_norm_data, 3),
        }

    # -------------------------------------------------------------
    # 主循环
    # -------------------------------------------------------------
    for idx, case in enumerate(current_tasks):
        mach, aoa = case["Ma"], case["AoA"]
        target_norm = ([mach, aoa] - l_min[0]) / (l_max[0] - l_min[0] + 1e-8)

        # ---- 模型预测 ----
        pred_field_norm = sample_flow_field_scheme_b(
            model, device,
            torch.tensor(grid_x_high, dtype=torch.float32, device=device).unsqueeze(0),
            torch.tensor(grid_y_high, dtype=torch.float32, device=device).unsqueeze(0),
            torch.tensor(target_norm, dtype=torch.float32, device=device).unsqueeze(0)
        )

        # ---- 真值检索 ----
        if os.path.exists(train_data_path):
            train_data = np.load(train_data_path)
            dists_train = np.linalg.norm(train_data['y'] - target_norm, axis=1)
            dists_test  = np.linalg.norm(test_data['y']  - target_norm, axis=1)
            min_tr, idx_tr = np.min(dists_train), np.argmin(dists_train)
            min_te, idx_te = np.min(dists_test),  np.argmin(dists_test)
            if min_tr <= min_te:
                gt_field_norm, matched_labels, tag = train_data['x'][idx_tr], train_data['y'][idx_tr], "Train"
            else:
                gt_field_norm, matched_labels, tag = test_data['x'][idx_te],  test_data['y'][idx_te],  "Test"
        else:
            dists = np.linalg.norm(test_data['y'] - target_norm, axis=1)
            best_match_idx = np.argmin(dists)
            gt_field_norm, matched_labels, tag = test_data['x'][best_match_idx], test_data['y'][best_match_idx], "Fallback Test"

        matched_phys = matched_labels * (l_max[0] - l_min[0] + 1e-8) + l_min[0]

        # ---- 反归一化：预测场 & 真值场 ----
        pd = denormalize_all(pred_field_norm)
        gt = denormalize_all(gt_field_norm)

        # ---- 提取 Cp 并积分 Cl / Cd ----
        x_up, cp_up_gt, x_lo, cp_lo_gt, cl_gt, cdp_gt, p_inf_gt, q_inf_gt = extract_raw_cp_and_forces(
            grid_x_high, grid_y_high, gt['p'],
            rho_field=gt['rho'], u_field=gt['u'], v_field=gt['v'],
            num_layers=4
        )
        _, cp_up_pd, _, cp_lo_pd, cl_pd, cdp_pd, p_inf_pd, q_inf_pd = extract_raw_cp_and_forces(
            grid_x_high, grid_y_high, pd['p'],
            rho_field=pd['rho'], u_field=pd['u'], v_field=pd['v'],
            num_layers=4
        )

        # ---- 终端日志 ----
        print(f"\n📊 [工况力学核算报告] Mach={mach:.3f}, AoA={aoa:.1f}° ({tag}匹配):")
        print(f"    -> 来流动压 q_inf: GT = {q_inf_gt:.4f},  Pred = {q_inf_pd:.4f}")
        print(f"    -> 真实值 (CFD)  : Cl = {cl_gt:.4f}, Cdp = {cdp_gt:.5f}")
        print(f"    -> 预测值 (Model): Cl = {cl_pd:.4f}, Cdp = {cdp_pd:.5f}")
        print(f"    -> 绝对偏差值    : ΔCl = {abs(cl_pd - cl_gt):.4f}, ΔCdp = {abs(cdp_pd - cdp_gt):.5f}")

        # ---- 绘图 ----
        ax = axes[idx]
        ax.plot(x_up, cp_up_gt, 'k--', label=f'GT CFD Upper (Cl={cl_gt:.3f})', alpha=0.7)
        ax.plot(x_lo, cp_lo_gt, 'k:',  label=f'GT CFD Lower', alpha=0.7)
        ax.plot(x_up, cp_up_pd, color='crimson',    linewidth=2.0,
                label=f'Pred Upper (Cl={cl_pd:.3f})')
        ax.plot(x_lo, cp_lo_pd, color='dodgerblue', linewidth=2.0,
                label='Pred Lower')

        ax.invert_yaxis()
        ax.set_title(
            f"Ma = {mach:.3f}, AoA = {aoa:.1f}°\n"
            f"Model: Cl={cl_pd:.3f}, Cd={cdp_pd:.5f} | "
            f"GT: Cl={cl_gt:.3f}, Cd={cdp_gt:.5f}",
            fontsize=8, fontweight='semibold'
        )
        ax.set_xlabel("Normalized Chord Length ($X/C$)", fontsize=9)
        ax.set_ylabel("Pressure Coefficient ($C_p$)", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    print("\n🎉 气动力系数积分与可视化矩阵全部清算完毕！正在呼出图像...")
    plt.show()