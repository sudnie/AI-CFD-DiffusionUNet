#usr/bin/python3
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# 🌟 1. 引入重构后的 4 层自适应高保真网络架构
from model_utils import HighResDiffusionUNet

# ==========================================
# 🌟 基于 x0-prediction 的正统反向采样器（高保真变尺寸自适应）
# ==========================================
@torch.no_grad()
def sample_flow_field_scheme_b(model, device, grid_x, grid_y, target_cond, timesteps=1000):
    model.eval()
    beta = torch.linspace(1e-4, 0.02, timesteps).to(device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    
    # 🎯 动态自适应原始大网格的高宽尺寸，直接在原生空间中注入噪声演进
    h, w = grid_x.shape[1], grid_x.shape[2]
    x_t = torch.randn((1, 4, h, w), device=device)
    
    for t_idx in reversed(range(timesteps)):
        t = torch.full((1,), t_idx, device=device, dtype=torch.long)
        # 网络内部会自动进行 8的倍数 Padding (60x301 -> 64x304) 并自动无损剪裁输出
        x0_pred = model(x_t, grid_x, grid_y, t, target_cond)
        ab_t = alpha_bar[t_idx]
        
        if t_idx > 0:
            ab_t_prev = alpha_bar[t_idx - 1]
            weight_x0 = torch.sqrt(ab_t_prev) * beta[t_idx] / (1.0 - ab_t)
            weight_xt = torch.sqrt(alpha[t_idx]) * (1.0 - ab_t_prev) / (1.0 - ab_t)
            var = (1.0 - ab_t_prev) / (1.0 - ab_t) * beta[t_idx]
            
            x_t = weight_x0 * x0_pred + weight_xt * x_t + torch.sqrt(var) * torch.randn_like(x_t)
        else:
            x_t = x0_pred
            
    return x_t

# ==========================================
# 2. 多模型检索与高保真对比主程序
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动 (60x301) 高保真多模型联合气动对齐评估矩阵: {device}")
    
    # 🌟【测试案例选择】输入你期望观测的目标物理参数
    TARGET_MACH = 0.475
    TARGET_AOA = 6
    
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else "."
    
    # 🎯 全部数据路径、系数包切换至 highres 阵列
    norm_path = os.path.abspath(os.path.join(current_dir, "../Results/normalization_factors_highres.npz"))
    train_data_path = os.path.abspath(os.path.join(current_dir, "../Results/HighRes_airfoil_train.npz"))
    test_data_path = os.path.abspath(os.path.join(current_dir, "../Results/HighRes_airfoil_test.npz"))
    
    # 自动扫描并建立高保真（HighRes）权重字典
    COMPARE_MODELS = {}
    for i in range(1000, 5001, 1000):
        COMPARE_MODELS[f"Epoch {i}"] = os.path.abspath(os.path.join(current_dir, f"../Results/airfoil_diffusion_highres_ep{i}.pth"))
        
    # 加载高保真基础数据
    train_data = np.load(train_data_path)
    test_data = np.load(test_data_path)
    norm_factors = np.load(norm_path)
    
    f_min, f_max = norm_factors['fields_min'], norm_factors['fields_max']
    l_min, l_max = norm_factors['label_min'], norm_factors['label_max']
    grid_x_raw, grid_y_raw = train_data['grid_x'], train_data['grid_y'] # [60, 301]
    
    # 🌟🌟🌟 全局最近邻检索真场（双库全量无损扫描） 🌟🌟🌟
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
    print(f"🔍 真正场源匹配成功: 【{data_source}】 #内部物理索引 {idx}")
    print(f"📊 [物理对齐验证] 最终抓取到的气动参考工况为：")
    print(f"    -> 真实世界马赫数 Mach: {cond_phys[0]:.4f} (期望: {TARGET_MACH:.4f})")
    print(f"    -> 真实世界攻角 AoA:    {cond_phys[1]:.4f}° (期望: {TARGET_AOA:.1f}°)")
    
    # 格式化特征输入
    cond_tensor = torch.tensor(target_norm, dtype=torch.float32, device=device).unsqueeze(0)
    grid_x_tensor = torch.tensor(grid_x_raw, dtype=torch.float32, device=device).unsqueeze(0)
    grid_y_tensor = torch.tensor(grid_y_raw, dtype=torch.float32, device=device).unsqueeze(0)
    
    # 压力与速度的高精反归一化物理还原闭包
    def denormalize(field_tensor):
        arr = field_tensor.cpu().numpy() if isinstance(field_tensor, torch.Tensor) else field_tensor
        if len(arr.shape) == 4: arr = arr[0]
        phys_0_1 = (arr + 1.0) / 2.0
        f_min_c = f_min[0, :, 0, 0].reshape(4, 1, 1)
        f_max_c = f_max[0, :, 0, 0].reshape(4, 1, 1)
        return phys_0_1 * (f_max_c - f_min_c) + f_min_c
    
    gt_phys = denormalize(torch.tensor(gt_fields_norm))
    
    # 🌟 动态循环推理所有指定的高清权重
    model_predictions = {}
    model = HighResDiffusionUNet().to(device) # 使用 4 层高保真 UNet
    
    for model_name, weights_key_path in COMPARE_MODELS.items():
        if not os.path.exists(weights_key_path):
            print(f"⚠️ 跳过模型 [{model_name}]: 找不到文件 {weights_key_path}")
            continue
            
        print(f"🌪️ 正在加载并运行高保真模型 [{model_name}] 的反向降噪生成...")
        model.load_state_dict(torch.load(weights_key_path, map_location=device, weights_only=True))
        
        # 移除了固定的 img_size，自适应生成 (60, 301) 的高保真精细流场
        pred_field_norm = sample_flow_field_scheme_b(model, device, grid_x_tensor, grid_y_tensor, cond_tensor)
        pred_phys = denormalize(pred_field_norm)
        
        model_predictions[model_name] = {
            "phys": pred_phys,
            "error": np.abs(gt_phys - pred_phys)
        }

    # ==========================================
    # 3. 🎨 动态网格可视化大图绘制 (横向对比矩阵)
    # ==========================================
    active_models = list(model_predictions.keys())
    num_models = len(active_models)
    
    if num_models == 0:
        raise RuntimeError("❌ 没有成功加载任何模型权重，请检查路径！")
        
    print(f"🎨 正在绘制高保真横向矩阵大图 (1个真场 + {num_models}个模型对比)...")
    
    # 严格对齐 4通道中 [U, V, Rho, P] 的物理分布 (0-U速度, 3-Pressure压力)
    target_channels = [0, 3] 
    channel_names = ["U Velocity", "Pressure"]
    
    fig, axes = plt.subplots(4, 1 + num_models, figsize=(4 * (1 + num_models), 14))
    fig.suptitle(f"High-Resolution (60x301) Multi-Model Aerodynamic Field Matrix\nMach={TARGET_MACH:.3f}, AoA={TARGET_AOA:.1f}°", fontsize=14, fontweight='bold')
    
    levels = 50
    cmap, err_cmap = 'jet', 'magma'
    
    for c_idx, ch_num in enumerate(target_channels):
        row_offset = c_idx * 2
        
        # --- 1. 绘制第一列：Ground Truth 真场 ---
        ax_gt = axes[row_offset, 0]
        ax_gt.set_title(f"GT - {channel_names[c_idx]}", fontsize=10, fontweight='bold')
        im_gt = ax_gt.contourf(grid_x_raw, grid_y_raw, gt_phys[ch_num], levels=levels, cmap=cmap)
        fig.colorbar(im_gt, ax=ax_gt)
        
        axes[row_offset + 1, 0].axis('off')
        axes[row_offset + 1, 0].text(0.5, 0.5, f"GT Reference\n(From {data_source.split()[0]})\n[60x301 Native]", ha='center', va='center', fontsize=10, color='gray', fontweight='bold')
        
        # --- 2. 循环绘制后续列：各个高保真模型的预测场与绝对误差场 ---
        for m_col, model_name in enumerate(active_models, start=1):
            pred_data = model_predictions[model_name]["phys"][ch_num]
            error_data = model_predictions[model_name]["error"][ch_num]
            
            # 预测场云图
            ax_pred = axes[row_offset, m_col]
            ax_pred.set_title(f"{model_name}\nPred", fontsize=10)
            im_pred = ax_pred.contourf(grid_x_raw, grid_y_raw, pred_data, levels=levels, cmap=cmap)
            fig.colorbar(im_pred, ax=ax_pred)
            
            # 绝对误差场云图
            ax_err = axes[row_offset + 1, m_col]
            ax_err.set_title(f"{model_name}\nAbs Error", fontsize=10)
            im_err = ax_err.contourf(grid_x_raw, grid_y_raw, error_data, levels=levels, cmap=err_cmap)
            fig.colorbar(im_err, ax=ax_err)

    # 刚性几何网格收尾
    for ax in axes.flatten():
        if ax.axison:
            ax.axis('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.tight_layout()
    save_path = os.path.abspath(os.path.join(current_dir, f"../Results/HighRes_Multi_Model_Comparison_Ma{TARGET_MACH:.3f}.png"))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🎉 高保真横向联合评估矩阵云图已成功导出至: {save_path}")
    plt.show()