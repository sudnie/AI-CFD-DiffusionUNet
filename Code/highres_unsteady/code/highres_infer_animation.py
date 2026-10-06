#!/usr/bin/python3#-*- coding: utf-8 -*-
"""
基于 Matplotlib 渲染引擎 + 轻量 HTML 前端控制器的 CFD 流场动画生成器
- 原生 C-Grid 三角剖分 (matplotlib.tri.Triangulation) 渲染，绝对保持物理几何与数值精度
- 后端并行/快速导出高精渲染帧，前端 JS 负责流畅交互，彻底摆脱 Plotly 渲染 Bug
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from tqdm import tqdm
from model_utils import DiffusionUNet

#==========================================#0. 参数配置#==========================================DATA_SUFFIX = "down1"            # "full" 或 "down4" 或 "down1"
USE_MIXED_DATASET = True         # True: train+test 混合按时间排序, False: 仅使用 DATASET_TYPE
DATASET_TYPE = "test"

USE_TIME_RANGE = True            # True: 按物理时间筛选, False: 按 Index 筛选
START_TIME = 120.0
END_TIME = 170.0
START_INDEX = 0
END_INDEX = 20

DDIM_STEPS = 50
BATCH_SIZE = 10
EXPORT_DPI = 150                 # 导出的 PNG 图像分辨率 (150-200 即可兼顾画质与生成速度)

#==========================================#1. 辅助函数：빠른 DDIM 샘플링 & 反归一化#==========================================@torch.no_grad()
def sample_ddim_batch(model, device, grid_x, grid_y, cond_batch, total_timesteps=1000, ddim_steps=50):
    model.eval()
    bsz = cond_batch.shape[0]
    H, W = grid_x.shape[-2], grid_x.shape[-1]
    beta = torch.linspace(1e-4, 0.02, total_timesteps).to(device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    times = torch.linspace(total_timesteps - 1, 0, ddim_steps, dtype=torch.long, device=device)
    x_t = torch.randn((bsz, 4, H, W), device=device)
    grid_x_b = grid_x.repeat(bsz, 1, 1, 1)
    grid_y_b = grid_y.repeat(bsz, 1, 1, 1)
    for i in range(len(times)):
        t_idx = times[i]
        t = torch.full((bsz,), t_idx, device=device, dtype=torch.long)
        x0_pred = model(x_t, grid_x_b, grid_y_b, t, cond_batch)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        if i == len(times) - 1:
            x_t = x0_pred
            break
        t_next = times[i + 1]
        ab_t = alpha_bar[t_idx]
        ab_next = alpha_bar[t_next]
        eps = (x_t - torch.sqrt(ab_t) * x0_pred) / (torch.sqrt(1.0 - ab_t) + 1e-8)
        x_t = torch.sqrt(ab_next) * x0_pred + torch.sqrt(1.0 - ab_next) * eps
    return x_t.cpu().numpy()

def denormalize(field_norm, f_min, f_max):
    arr = np.array(field_norm)
    phys_0_1 = (arr + 1.0) / 2.0
    f_min_c = f_min.reshape(1, 4, 1, 1)
    f_max_c = f_max.reshape(1, 4, 1, 1)
    return phys_0_1 * (f_max_c - f_min_c) + f_min_c

def load_and_select(data_path, norm_path, time_range=None, indices_range=None, use_time=True):
    data = np.load(data_path)
    norm = np.load(norm_path)
    f_min, f_max = norm["fields_min"], norm["fields_max"]
    l_min, l_max = float(np.squeeze(norm["label_min"])), float(np.squeeze(norm["label_max"]))

    x_norm = data["x"]
    y_norm = data["y"]
    grid_x = data["grid_x"]
    grid_y = data["grid_y"]

    if use_time:
        y_phys = (y_norm * (l_max - l_min) + l_min).flatten()
        start, end = time_range
        idx = np.where((y_phys >= start) & (y_phys <= end))[0]
        if len(idx) == 0:
            return None, None, None, None, None, None, None, None
        idx = idx[np.argsort(y_phys[idx])]
        times = y_phys[idx]
    else:
        s, e = indices_range
        idx = list(range(s, e + 1))
        y_phys = y_norm[idx] * (l_max - l_min) + l_min
        times = y_phys.flatten()

    return x_norm[idx], times, grid_x, grid_y, f_min, f_max, l_min, l_max

#==========================================#2. 主流程#==========================================if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🌟 启动 Matplotlib + HTML 前端渲染引擎 | 设备: {device}")

    current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
    results_dir = os.path.abspath(os.path.join(current_dir, "../Results"))

    #--- 模型与权重加载 ---    weights_candidates = [
        os.path.join(results_dir, f"airfoil_diffusion_cgrid_{DATA_SUFFIX}_final.pth"),
        os.path.join(results_dir, f"airfoil_diffusion_cgrid_{DATA_SUFFIX}_ep3000.pth"),
        os.path.join(results_dir, f"airfoil_diffusion_cgrid_{DATA_SUFFIX}_ep2000.pth"),
        os.path.join(results_dir, f"airfoil_diffusion_cgrid_{DATA_SUFFIX}_ep1000.pth"),
    ]
    weights_path = next((w for w in weights_candidates if os.path.exists(w)), None)
    if weights_path is None:
        raise FileNotFoundError(f"❌ 未找到权重文件 (DATA_SUFFIX: {DATA_SUFFIX})")

    norm_path = os.path.join(results_dir, f"normalization_factors_{DATA_SUFFIX}.npz")
    train_path = os.path.join(results_dir, f"Diffusion_airfoil_unsteady_{DATA_SUFFIX}_train.npz")
    test_path = os.path.join(results_dir, f"Diffusion_airfoil_unsteady_{DATA_SUFFIX}_test.npz")

    model = DiffusionUNet(flow_ch=4, coord_ch=2, cond_dim=128, base_ch=48).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    print(f"✅ 模型权重加载成功: {weights_path}")

    #--- 数据源读取 ---    if USE_MIXED_DATASET:
        tr_x, tr_t, tr_gx, tr_gy, f_min, f_max, l_min, l_max = load_and_select(
            train_path, norm_path, (START_TIME, END_TIME), (START_INDEX, END_INDEX), USE_TIME_RANGE
        )
        te_x, te_t, te_gx, te_gy, _, _, _, _ = load_and_select(
            test_path, norm_path, (START_TIME, END_TIME), (START_INDEX, END_INDEX), USE_TIME_RANGE
        )
        grid_x_raw = tr_gx if tr_gx is not None else te_gx
        grid_y_raw = tr_gy if tr_gy is not None else te_gy

        frames = []
        if tr_x is not None:
            for i in range(len(tr_x)): frames.append((tr_x[i], tr_t[i], 'train'))
        if te_x is not None:
            for i in range(len(te_x)): frames.append((te_x[i], te_t[i], 'test'))
        frames.sort(key=lambda item: item[1])
        x_list = [f[0] for f in frames]
        time_list = [f[1] for f in frames]
        source_list = [f[2] for f in frames]
    else:
        d_path = os.path.join(results_dir, f"Diffusion_airfoil_unsteady_{DATA_SUFFIX}_{DATASET_TYPE}.npz")
        x_np, times, grid_x_raw, grid_y_raw, f_min, f_max, l_min, l_max = load_and_select(
            d_path, norm_path, (START_TIME, END_TIME), (START_INDEX, END_INDEX), USE_TIME_RANGE
        )
        x_list, time_list = x_np, times
        source_list = [DATASET_TYPE] * len(x_list)

    num_samples = len(x_list)
    print(f"📦 로드 완료: 共 {num_samples} 个时间步帧")

    #--- DDIM 批次推理 ---    grid_x_tensor = torch.tensor(grid_x_raw, dtype=torch.float32, device=device)
    grid_y_tensor = torch.tensor(grid_y_raw, dtype=torch.float32, device=device)
    grid_x_tensor = 2.0 * (grid_x_tensor - grid_x_tensor.min()) / (grid_x_tensor.max() - grid_x_tensor.min() + 1e-8) - 1.0
    grid_y_tensor = 2.0 * (grid_y_tensor - grid_y_tensor.min()) / (grid_y_tensor.max() - grid_y_tensor.min() + 1e-8) - 1.0
    grid_x_tensor = grid_x_tensor.unsqueeze(0).unsqueeze(0)
    grid_y_tensor = grid_y_tensor.unsqueeze(0).unsqueeze(0)

    gt_u_list = []
    pred_u_list = []
    err_u_list = []
    labels_list = []

    print("🔮 执行模型 DDIM 추론...")
    for start in tqdm(range(0, num_samples, BATCH_SIZE), desc="Inference"):
        end = min(start + BATCH_SIZE, num_samples)
        b_indices = list(range(start, end))
        batch_x_norm = np.stack([x_list[i] for i in b_indices])
        batch_times = [float(time_list[i]) for i in b_indices]
        batch_sources = [source_list[i] for i in b_indices]

        cond_vals = [(t - l_min) / (l_max - l_min + 1e-8) for t in batch_times]
        cond_tensor = torch.tensor(cond_vals, dtype=torch.float32, device=device).unsqueeze(1)

        pred_norm = sample_ddim_batch(model, device, grid_x_tensor, grid_y_tensor, cond_tensor, ddim_steps=DDIM_STEPS)

        for j in range(len(b_indices)):
            gt_phys = denormalize(batch_x_norm[j], f_min, f_max)[0]
            pred_phys = denormalize(pred_norm[j], f_min, f_max)[0]

            gt_u = gt_phys[1].flatten()
            pred_u = pred_phys[1].flatten()
            err_u = np.abs(gt_u - pred_u)

            gt_u_list.append(gt_u)
            pred_u_list.append(pred_u)
            err_u_list.append(err_u)
            labels_list.append(f"{batch_sources[j].upper()} (t={batch_times[j]:.1f}s)")

    #==========================================    #3. 预计算 C-Grid 三角网格拓扑与颜色范围    #==========================================    print("⚡ Build C-Grid 原始网格二维三角剖分拓扑...")
    pts_x = grid_x_raw.flatten()
    pts_y = grid_y_raw.flatten()
    triangulation = tri.Triangulation(pts_x, pts_y)

    wall_x = grid_x_raw[0, :]
    wall_y = grid_y_raw[0, :]

    u_min = min(np.min(gt_u_list), np.min(pred_u_list))
    u_max = max(np.max(gt_u_list), np.max(pred_u_list))
    err_max = np.max(err_u_list)

#==========================================    #4. 后端静态高清帧批量导出 (已修改放大范围)    #==========================================    frames_dir = os.path.join(results_dir, "cfd_frames_cache")
    os.makedirs(frames_dir, exist_ok=True)

    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=EXPORT_DPI)
    fig.subplots_adjust(wspace=0.22, bottom=0.12, top=0.88)

    titles = ["GT U-Velocity (m/s)", "Pred U-Velocity (m/s)", "Absolute Error (U)"]

    #初始化底层 tripcolor 图层    tpc_gt = axes[0].tripcolor(triangulation, gt_u_list[0], cmap='viridis', vmin=u_min, vmax=u_max, shading='gouraud')
    tpc_pred = axes[1].tripcolor(triangulation, pred_u_list[0], cmap='viridis', vmin=u_min, vmax=u_max, shading='gouraud')
    tpc_err = axes[2].tripcolor(triangulation, err_u_list[0], cmap='hot', vmin=0, vmax=err_max, shading='gouraud')

    cbar1 = fig.colorbar(tpc_gt, ax=[axes[0], axes[1]], orientation='vertical', fraction=0.02, pad=0.02)
    cbar1.set_label('Velocity U (m/s)', fontsize=10, fontweight='bold')

    cbar2 = fig.colorbar(tpc_err, ax=axes[2], orientation='vertical', fraction=0.04, pad=0.02)
    cbar2.set_label('Absolute Error', fontsize=10, fontweight='bold')

    #定义目标放大区域：[-1, 1]    CROP_X_MIN, CROP_X_MAX = -1, 5
    CROP_Y_MIN, CROP_Y_MAX = -1.0, 1.0

    for i, ax in enumerate(axes):
        ax.plot(wall_x, wall_y, 'k-', linewidth=1.5)
        ax.set_title(titles[i], fontsize=12, fontweight='bold')
        ax.set_xlabel('X (m)')
        if i == 0:
            ax.set_ylabel('Y (m)')
        
        #核心修改：锁定 X、Y 范围为 [-1, 1]，并保持 1:1 物理真实比例        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(CROP_X_MIN, CROP_X_MAX)
        ax.set_ylim(CROP_Y_MIN, CROP_Y_MAX)
        ax.grid(True, linestyle='--', alpha=0.3)

    suptitle = fig.suptitle("", fontsize=14, fontweight='bold')

    print("🎨 正在导出近壁面放大区域 ([-1, 1]) 的高清流场帧...")
    img_filenames = []
    for k in tqdm(range(num_samples), desc="Exporting PNGs"):
        tpc_gt.set_array(gt_u_list[k])
        tpc_pred.set_array(pred_u_list[k])
        tpc_err.set_array(err_u_list[k])
        
        suptitle.set_text(f"Airfoil Flow Field Dynamics (Zoomed) | Time: {time_list[k]:.2f}s [{source_list[k].upper()}]")
        
        img_name = f"frame_{k:04d}.png"
        img_path = os.path.join(frames_dir, img_name)
        fig.savefig(img_path, dpi=EXPORT_DPI, bbox_inches='tight')
        img_filenames.append(img_name)

    plt.close()

    #==========================================    #5. 生成轻量 HTML/JS 前端控制网页    #==========================================    player_html_path = os.path.join(results_dir, f"cfd_flow_player_{DATA_SUFFIX}.html")

    html_code = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>Airfoil CFD Flow Dynamics Player</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .card {{
            background-color: #1e293b;
            border-radius: 12px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
            padding: 20px;
            max-width: 1400px;
            width: 100%;
            box-sizing: border-box;
        }}
        .img-container {{
            width: 100%;
            background-color: #020617;
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 400px;
            border: 1px solid #334155;
        }}
        img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .controls-panel {{
            margin-top: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 15px;
            background-color: #0f172a;
            padding: 12px 20px;
            border-radius: 8px;
        }}
        .btn {{
            background-color: #3b82f6;
            color: white;
            border: none;
            padding: 8px 18px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
            min-width: 90px;
        }}
        .btn:hover {{ background-color: #2563eb; }}
        .btn-pause {{ background-color: #ef4444; }}
        .btn-pause:hover {{ background-color: #dc2626; }}
        
        .slider-wrapper {{
            flex-grow: 1;
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        input[type=range] {{
            flex-grow: 1;
            height: 6px;
            border-radius: 3px;
            background: #475569;
            outline: none;
            cursor: pointer;
        }}
        .time-badge {{
            font-family: monospace;
            font-size: 14px;
            background-color: #334155;
            padding: 4px 10px;
            border-radius: 4px;
            color: #38bdf8;
            white-space: nowrap;
        }}
        .hint {{
            margin-top: 10px;
            font-size: 12px;
            color: #94a3b8;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="card">
        <div class="img-container">
            <img id="flowViewer" src="cfd_frames_cache/{img_filenames[0]}" alt="Flow Field Frame">
        </div>
        <div class="controls-panel">
            <button id="playBtn" class="btn" onclick="togglePlay()">▶ Play</button>
            <div class="slider-wrapper">
                <input type="range" id="timeSlider" min="0" max="{num_samples - 1}" value="0" oninput="onSliderInput(this.value)">
                <span id="timeBadge" class="time-badge">{labels_list[0]}</span>
            </div>
        </div>
        <div class="hint">💡 提示：按键盘 <kbd>←</kbd> <kbd>→</kbd> 方向键可单帧微调</div>
    </div>

    <script>
        const frames = {img_filenames};
        const labels = {labels_list};
        let currentIndex = 0;
        let timer = null;

        const viewer = document.getElementById('flowViewer');
        const slider = document.getElementById('timeSlider');
        const badge = document.getElementById('timeBadge');
        const playBtn = document.getElementById('playBtn');

        function setFrame(idx) {{
            currentIndex = parseInt(idx);
            viewer.src = 'cfd_frames_cache/' + frames[currentIndex];
            slider.value = currentIndex;
            badge.innerText = labels[currentIndex];
        }}

        function onSliderInput(val) {{
            if (timer) pause();
            setFrame(val);
        }}

        function play() {{
            playBtn.innerText = "❚❚ Pause";
            playBtn.className = "btn btn-pause";
            timer = setInterval(() => {{
                currentIndex = (currentIndex + 1) % frames.length;
                setFrame(currentIndex);
            }}, 100);
        }}

        function pause() {{
            playBtn.innerText = "▶ Play";
            playBtn.className = "btn";
            if (timer) {{
                clearInterval(timer);
                timer = null;
            }}
        }}

        function togglePlay() {{
            if (timer) pause();
            else play();
        }}

        // 绑定键盘左右箭头按键逐帧控制
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowLeft') {{
                pause();
                let nextIdx = (currentIndex - 1 + frames.length) % frames.length;
                setFrame(nextIdx);
            }} else if (e.key === 'ArrowRight') {{
                pause();
                let nextIdx = (currentIndex + 1) % frames.length;
                setFrame(nextIdx);
            }} else if (e.key === ' ') {{
                togglePlay();
                e.preventDefault();
            }}
        }});
    </script>
</body>
</html>
"""

    with open(player_html_path, "w", encoding="utf-8") as f:
        f.write(html_code)

    print(f"\n🎉 处理完成！请在浏览器中直接打开 HTML 文件查看：\n👉 {player_html_path}")