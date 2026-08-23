#usr/bin/python3
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ==========================================
# 1. 基础工具模块
# ==========================================
def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

def spatial_gradient(tensor):
    """有限差分空间导数"""
    grad_x = F.pad(tensor[:, :, :, 1:] - tensor[:, :, :, :-1], (0, 1, 0, 0))
    grad_y = F.pad(tensor[:, :, 1:, :] - tensor[:, :, :-1, :], (0, 0, 0, 1))
    return grad_x, grad_y

# ==========================================
# 2. 神经网络核心组件 (支持 AdaIN 开关)
# ==========================================
class AdaIN(nn.Module):
    def __init__(self, cond_dim, channels):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels, affine=False)
        self.fc = nn.Linear(cond_dim, channels * 2)

    def forward(self, x, cond):
        x_norm = self.instance_norm(x)
        gamma_beta = self.fc(cond).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        return (1 + gamma) * x_norm + beta

class ConfigurableConditionalResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, use_adain=True):
        super().__init__()
        self.use_adain = use_adain
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # 🎛️ 开关：根据 use_adain 决定使用 AdaIN 还是普通的 InstanceNorm + 特征相加
        if self.use_adain:
            self.adain1 = AdaIN(cond_dim, out_channels)
            self.adain2 = AdaIN(cond_dim, out_channels)
        else:
            self.norm1 = nn.InstanceNorm2d(out_channels)
            self.norm2 = nn.InstanceNorm2d(out_channels)
            self.cond_proj1 = nn.Linear(cond_dim, out_channels)
            self.cond_proj2 = nn.Linear(cond_dim, out_channels)
            
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(x)
        
        # 🎛️ 逻辑分支
        if self.use_adain:
            h = self.adain1(h, cond)
        else:
            h = self.norm1(h)
            h = h + self.cond_proj1(cond).unsqueeze(-1).unsqueeze(-1)
            
        h = self.act(h)
        h = self.conv2(h)
        
        # 🎛️ 逻辑分支
        if self.use_adain:
            h = self.adain2(h, cond)
        else:
            h = self.norm2(h)
            h = h + self.cond_proj2(cond).unsqueeze(-1).unsqueeze(-1)
            
        return h + self.shortcut(x)

class ConfigurableDiffusionUNet(nn.Module):
    """
    带有架构开关的 UNet，方便进行消融实验验证模块有效性
    """
    def __init__(self, flow_ch=4, coord_ch=2, cond_dim=128, use_adain=True):
        super().__init__()
        self.use_adain = use_adain
        self.init_conv = nn.Conv2d(flow_ch + coord_ch, 64, 3, padding=1)
        
        self.time_mlp = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.physics_mlp = nn.Sequential(nn.Linear(2, cond_dim // 2), nn.SiLU(), nn.Linear(cond_dim // 2, cond_dim))

        # 传入 use_adain 开关参数
        self.down1 = ConfigurableConditionalResBlock(64, 64, cond_dim, use_adain=use_adain)
        self.down2 = ConfigurableConditionalResBlock(64, 128, cond_dim, use_adain=use_adain)
        self.down3 = ConfigurableConditionalResBlock(128, 256, cond_dim, use_adain=use_adain)
        
        self.up1 = ConfigurableConditionalResBlock(256 + 128, 128, cond_dim, use_adain=use_adain)
        self.up2 = ConfigurableConditionalResBlock(128 + 64, 64, cond_dim, use_adain=use_adain)
        self.up3 = ConfigurableConditionalResBlock(64 + 64, 64, cond_dim, use_adain=use_adain)
        
        self.final_conv = nn.Conv2d(64, flow_ch, 1)

    def forward(self, x_t, grid_x, grid_y, t, physics_cond):
        x = torch.cat([x_t, grid_x.unsqueeze(1), grid_y.unsqueeze(1)], dim=1) 
        cond = self.time_mlp(get_timestep_embedding(t, 128)) + self.physics_mlp(physics_cond)
        
        x0 = self.init_conv(x)
        d1 = self.down1(x0, cond)
        d2 = self.down2(F.max_pool2d(d1, 2), cond)
        d3 = self.down3(F.max_pool2d(d2, 2), cond)
        
        u1 = self.up1(torch.cat([F.interpolate(d3, scale_factor=2, mode='nearest'), d2], dim=1), cond)
        u2 = self.up2(torch.cat([F.interpolate(u1, scale_factor=2, mode='nearest'), d1], dim=1), cond)
        u3 = self.up3(torch.cat([u2, x0], dim=1), cond)
        
        return self.final_conv(u3)

# ==========================================
# 3. 可配置的物理启发损失函数 (PI-Loss)
# ==========================================
def configurable_physics_loss(pred_noise, true_noise, x0_pred, x0_true, grid_x, grid_y, loss_config):
    """
    可插拔的物理损失函数，由 loss_config 字典控制各个约束项的开关
    """
    # 1. 基础噪声扩散损失 (始终保留)
    mse_noise_loss = F.mse_loss(pred_noise, true_noise)
    total_loss = mse_noise_loss
    loss_details = {"noise_loss": mse_noise_loss.item()}

    radius_sq = grid_x**2 + grid_y**2

    # 🎛️ 开关：直接通道空间约束 (Velocity & Pressure Direct Alignment)
    if loss_config.get("use_direct_loss", True):
        vel_direct_loss = F.l1_loss(x0_pred[:, 0:3], x0_true[:, 0:3])
        press_direct_loss = F.l1_loss(x0_pred[:, 3:4], x0_true[:, 3:4]) * 2.5 
        direct_loss = vel_direct_loss + press_direct_loss
        total_loss = total_loss + direct_loss
        loss_details["direct_loss"] = direct_loss.item()

    # 🎛️ 开关：空间一阶导数梯度约束 (近场防御)
    if loss_config.get("use_grad_loss", True):
        spatial_weight = torch.exp(-2.0 * radius_sq).unsqueeze(1).expand_as(x0_pred)
        pred_grad_x, pred_grad_y = spatial_gradient(x0_pred)
        true_grad_x, true_grad_y = spatial_gradient(x0_true)
        
        grad_loss = (F.mse_loss(pred_grad_x, true_grad_x, reduction='none') + \
                     F.mse_loss(pred_grad_y, true_grad_y, reduction='none')) * spatial_weight
        
        lambda_grad = loss_config.get("lambda_grad", 1.0)
        grad_loss_mean = grad_loss.mean() * lambda_grad
        total_loss = total_loss + grad_loss_mean
        loss_details["grad_loss"] = grad_loss_mean.item()

    # 🎛️ 开关：远场边界硬约束 (Far-field Constraint)
    if loss_config.get("use_far_field_loss", True):
        far_field_mask = (radius_sq > 0.64).float().unsqueeze(1).expand_as(x0_pred)
        far_field_loss = F.l1_loss(x0_pred * far_field_mask, x0_true * far_field_mask) * 4.0
        total_loss = total_loss + far_field_loss
        loss_details["far_field_loss"] = far_field_loss.item()

    # 🎛️ 开关：极值强对齐惩罚 (Extrema Alignment)
    if loss_config.get("use_extrema_loss", True):
        max_penalty = F.mse_loss(x0_pred.max(dim=-1)[0].max(dim=-1)[0], x0_true.max(dim=-1)[0].max(dim=-1)[0])
        min_penalty = F.mse_loss(x0_pred.min(dim=-1)[0].min(dim=-1)[0], x0_true.min(dim=-1)[0].min(dim=-1)[0])
        extrema_loss = 0.5 * (max_penalty + min_penalty)
        total_loss = total_loss + extrema_loss
        loss_details["extrema_loss"] = extrema_loss.item()

    return total_loss, loss_details

# ==========================================
# 4. 数据集与扩散辅助函数 (新增真实数据支持)
# ==========================================
class AirfoilVRAMDataset(Dataset):
    def __init__(self, npz_path, device):
        print(f"📦 正在加载全量数据至显存 [{device}]: {npz_path}")
        data = np.load(npz_path)
        self.fields = torch.tensor(data['x'], dtype=torch.float32, device=device)
        self.labels = torch.tensor(data['y'], dtype=torch.float32, device=device)
        grid_x = torch.tensor(data['grid_x'], dtype=torch.float32, device=device)
        grid_y = torch.tensor(data['grid_y'], dtype=torch.float32, device=device)
        
        # 坐标归一化到 [-1, 1] 供网络更好地提取特征
        self.grid_x = 2.0 * (grid_x - grid_x.min()) / (grid_x.max() - grid_x.min() + 1e-8) - 1.0
        self.grid_y = 2.0 * (grid_y - grid_y.min()) / (grid_y.max() - grid_y.min() + 1e-8) - 1.0

    def __len__(self):
        return len(self.fields)

    def __getitem__(self, idx):
        return self.fields[idx], self.grid_x, self.grid_y, self.labels[idx]

def add_noise_fast(x0, t, sqrt_ab_table, sqrt_1_minus_ab_table):
    """前向加噪过程"""
    noise = torch.randn_like(x0)
    x_t = sqrt_ab_table[t] * x0 + sqrt_1_minus_ab_table[t] * noise
    return x_t, noise

@torch.no_grad()
def sample_flow_field(model, device, grid_x, grid_y, target_cond, img_size=(64, 64), timesteps=1000):
    """反向去噪演进过程 (预测 x0 的 Scheme B 逻辑)"""
    model.eval()
    beta = torch.linspace(1e-4, 0.02, timesteps).to(device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    
    # 从纯高斯白噪声开始
    x_t = torch.randn((1, 4, img_size[0], img_size[1]), device=device)
    
    pbar = tqdm(reversed(range(timesteps)), desc="🌪️ 反向降噪演进中", total=timesteps, leave=False)
    for t_idx in pbar:
        t = torch.full((1,), t_idx, device=device, dtype=torch.long)
        
        # 预估出当前步下纯净的流场 x0
        x0_pred = model(x_t, grid_x, grid_y, t, target_cond)
        
        ab_t = alpha_bar[t_idx]
        if t_idx > 0:
            ab_t_prev = alpha_bar[t_idx - 1]
            
            # 郎之万重参数化推导
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
# 5. 真实数据训练与预测主程序
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 使用设备: {device}")
    
    # ---------------------------
    # ⚙️ 核心流程控制器：自由控制训练或预测
    # ---------------------------
    DO_TRAIN = True         # True: 执行训练过程, False: 跳过训练直接加载权重预测
    DO_PREDICT = True       # True: 训练完后（或直接加载）执行推理可视化过程
    EPOCHS = 2000            # 演示训练轮数 (实战时可改回 3200)
    BATCH_SIZE = 128
    
    # 路径配置（对齐您项目的原有目录结构）
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else "."
    train_npz_path = os.path.abspath(os.path.join(current_dir, "../Results/Diffusion_airfoil_train.npz"))
    norm_path = os.path.abspath(os.path.join(current_dir, "../Results/normalization_factors.npz"))
    weights_save_path = os.path.abspath(os.path.join(current_dir, "../Results/airfoil_diffusion_64_ep3200.pth"))
    
    # 🎛️ 定义当前您想跑的消融配置结构
    active_config = {
        "use_adain": True,  # 可以改为 False 测试纯净 CNN 融合效果
        "loss_config": {
            "use_direct_loss": False,
            "use_grad_loss": False,
            "use_far_field_loss": False,
            "use_extrema_loss": False,
            "lambda_grad": 1.0
        }
    }
    
    # 实例化网络
    model = ConfigurableDiffusionUNet(use_adain=active_config["use_adain"]).to(device)

    # ========================================
    # 🏃‍♂️ 模块A：接入真实流场数据集进行训练
    # ========================================
    if DO_TRAIN:
        if not os.path.exists(train_npz_path):
            print(f"\n❌ 找不到训练数据: {train_npz_path}")
            print("💡 请确保您的 Diffusion_airfoil_train.npz 文件放置在脚本上级目录的 Results 文件夹中！")
        else:
            print("\n🚀 正在载入真实数据集并启动训练引擎...")
            dataset = AirfoilVRAMDataset(train_npz_path, device=device)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
            scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
            
            num_timesteps = 1000
            beta = torch.linspace(1e-4, 0.02, num_timesteps).to(device)
            alpha_bar = torch.cumprod(1.0 - beta, dim=0)
            sqrt_ab_table = torch.sqrt(alpha_bar).view(-1, 1, 1, 1)
            sqrt_1_minus_ab_table = torch.sqrt(1.0 - alpha_bar + 1e-8).view(-1, 1, 1, 1)
            
            global_pbar = tqdm(range(EPOCHS), desc="训练进度")
            
            for epoch in global_pbar:
                model.train()
                epoch_loss = 0.0
                
                for flow, grid_x, grid_y, cond in dataloader:
                    # 1. 抽取时间步，极速加噪
                    t = torch.randint(0, num_timesteps, (flow.shape[0],), device=device).long()
                    x_t, true_noise = add_noise_fast(flow, t, sqrt_ab_table, sqrt_1_minus_ab_table)
                    
                    optimizer.zero_grad()
                    with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                        # 2. 网络前向推理
                        x0_pred = model(x_t, grid_x, grid_y, t, cond)
                        pred_noise = (x_t - sqrt_ab_table[t] * x0_pred) / sqrt_1_minus_ab_table[t]
                        
                        # 3. 动态计算可配置物理损失
                        loss, details = configurable_physics_loss(
                            pred_noise, true_noise, x0_pred, flow, grid_x, grid_y, active_config["loss_config"]
                        )
                    
                    # 4. 反向传播更新
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    epoch_loss += loss.item()
                    
                scheduler.step()
                avg_loss = epoch_loss / len(dataloader)
                global_pbar.set_postfix({"Avg_Loss": f"{avg_loss:.5f}", "LR": f"{optimizer.param_groups[0]['lr']:.2e}"})
            
            # 训练结束，安全固化网络参数
            os.makedirs(os.path.dirname(weights_save_path), exist_ok=True)
            torch.save(model.state_dict(), weights_save_path)
            print(f"🎉 训练顺利完工！模型已固化至: {weights_save_path}")

    # ========================================
    # 🎨 模块B：真实推理解析与流场可视化 (含归一化与误差计算)
    # ========================================
    if DO_PREDICT:
        if not os.path.exists(weights_save_path):
            print(f"\n❌ 推理失败：找不到权重文件 {weights_save_path}。请先设置 DO_TRAIN=True 训练一次。")
        elif not os.path.exists(train_npz_path):
            print(f"\n❌ 推理失败：需要借用 {train_npz_path} 提取测试所需的翼型基础网格坐标。")
        else:
            print("\n🖼️ 正在挂载网络权重，生成全新的物理预测流场...")
            model.load_state_dict(torch.load(weights_save_path, map_location=device))
            
            # 载入数据并挑选测试样例 (此处以索引 0 的数据为例)
            data = np.load(train_npz_path)
            test_idx = 0 
            
            grid_x_raw = data['grid_x']
            grid_y_raw = data['grid_y']
            
            ref_grid_x = torch.tensor(grid_x_raw, dtype=torch.float32, device=device)
            ref_grid_y = torch.tensor(grid_y_raw, dtype=torch.float32, device=device)
            norm_grid_x = 2.0 * (ref_grid_x - ref_grid_x.min()) / (ref_grid_x.max() - ref_grid_x.min() + 1e-8) - 1.0
            norm_grid_y = 2.0 * (ref_grid_y - ref_grid_y.min()) / (ref_grid_y.max() - ref_grid_y.min() + 1e-8) - 1.0
            
            # 抽取并格式化目标物理条件与 Ground Truth
            target_cond_norm = torch.tensor(data['y'][test_idx], dtype=torch.float32, device=device).unsqueeze(0)
            gt_field_norm = data['x'][test_idx]
            
            # 初始化反归一化模块
            if os.path.exists(norm_path):
                norm_factors = np.load(norm_path)
                f_min, f_max = norm_factors['fields_min'], norm_factors['fields_max']
                l_min, l_max = norm_factors['label_min'], norm_factors['label_max']
                
                cond_phys = data['y'][test_idx] * (l_max[0] - l_min[0] + 1e-8) + l_min[0]
                target_mach, target_aoa = cond_phys[0], cond_phys[1]
                
                def denormalize(field_tensor):
                    arr = field_tensor if isinstance(field_tensor, np.ndarray) else field_tensor.cpu().numpy()
                    if len(arr.shape) == 4: arr = arr[0]
                    phys_0_1 = (arr + 1.0) / 2.0
                    f_min_c = f_min[0, :, 0, 0].reshape(4, 1, 1)
                    f_max_c = f_max[0, :, 0, 0].reshape(4, 1, 1)
                    return phys_0_1 * (f_max_c - f_min_c) + f_min_c
            else:
                print("⚠️ 警告: 找不到 normalization_factors.npz，直接基于归一化空间计算物理场与误差。")
                target_mach, target_aoa = data['y'][test_idx][0], data['y'][test_idx][1]
                def denormalize(field_tensor):
                    arr = field_tensor if isinstance(field_tensor, np.ndarray) else field_tensor.cpu().numpy()
                    if len(arr.shape) == 4: arr = arr[0]
                    return arr
            
            print(f"📊 [正在评估气动工况] Mach: {target_mach:.4f}, AoA: {target_aoa:.1f}°")

            # 核心：跑满 1000 步的扩散逆向生成逻辑
            pred_field_norm = sample_flow_field(
                model, device, 
                norm_grid_x.unsqueeze(0), 
                norm_grid_y.unsqueeze(0), 
                target_cond_norm, 
                img_size=(64, 64)
            )
            
            # 执行反归一化，得到真正的物理场
            gt_phys = denormalize(gt_field_norm)
            pred_phys = denormalize(pred_field_norm)
            
            # ==========================================
            # 误差清算：计算单通道 L2 及全局 L2 (通道映射 0:U, 1:V, 2:Rho, 3:Press)
            # ==========================================
            eps = 1e-8
            ch_u, ch_v, ch_p = 0, 1, 3
            
            l2_error_u = np.linalg.norm(gt_phys[ch_u] - pred_phys[ch_u]) / (np.linalg.norm(gt_phys[ch_u]) + eps)
            l2_error_p = np.linalg.norm(gt_phys[ch_p] - pred_phys[ch_p]) / (np.linalg.norm(gt_phys[ch_p]) + eps)
            
            # 全局综合百分比误差 (Table 4)
            res_sq = np.sum((gt_phys[ch_u]-pred_phys[ch_u])**2) + np.sum((gt_phys[ch_v]-pred_phys[ch_v])**2) + np.sum((gt_phys[ch_p]-pred_phys[ch_p])**2)
            den_sq = np.sum(gt_phys[ch_u]**2) + np.sum(gt_phys[ch_v]**2) + np.sum(gt_phys[ch_p]**2)
            case_combined_l2_error = np.sqrt(res_sq / den_sq) * 100

            # --------- 2x2 制图展示 Ground Truth 对比 ---------
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f"Ablation Framework Output | Mach: {target_mach:.3f}, AoA: {target_aoa:.1f}°\nAdaIN Enabled: {active_config['use_adain']} | Global Combined L2 Error: {case_combined_l2_error:.3f}%", fontsize=15, fontweight='bold')
            
            # --- U-Velocity 对比 ---
            im0 = axes[0, 0].contourf(grid_x_raw, grid_y_raw, gt_phys[ch_u], levels=50, cmap='jet')
            axes[0, 0].set_title("Ground Truth - U Velocity", fontsize=12)
            fig.colorbar(im0, ax=axes[0, 0])
            axes[0, 0].axis('equal')
            
            im1 = axes[0, 1].contourf(grid_x_raw, grid_y_raw, pred_phys[ch_u], levels=50, cmap='jet')
            axes[0, 1].set_title(f"Predicted - U Velocity (L2 Error: {l2_error_u*100:.2f}%)", fontsize=12, color='darkred')
            fig.colorbar(im1, ax=axes[0, 1])
            axes[0, 1].axis('equal')
            
            # --- Pressure 对比 ---
            im2 = axes[1, 0].contourf(grid_x_raw, grid_y_raw, gt_phys[ch_p], levels=50, cmap='jet')
            axes[1, 0].set_title("Ground Truth - Pressure", fontsize=12)
            fig.colorbar(im2, ax=axes[1, 0])
            axes[1, 0].axis('equal')
            
            im3 = axes[1, 1].contourf(grid_x_raw, grid_y_raw, pred_phys[ch_p], levels=50, cmap='jet')
            axes[1, 1].set_title(f"Predicted - Pressure (L2 Error: {l2_error_p*100:.2f}%)", fontsize=12, color='darkred')
            fig.colorbar(im3, ax=axes[1, 1])
            axes[1, 1].axis('equal')
            
            plt.tight_layout()
            print("✅ 真实物理场对比渲染完成！误差指标已附带在图像中...")
            plt.show()