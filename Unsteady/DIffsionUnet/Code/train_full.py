#/usr/bin/python3
"""
全尺寸 145x689 C-grid DiffusionUNet 物理感知训练脚本
优化项：GPU Direct Batching + 动态尺寸物理加权 + AMP 混合精度
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from model_utils import DiffusionUNet


# ==========================================
# 1. GPU 内存直接 Batching 数据集
# ==========================================
class GPUDirectDataset:
    def __init__(self, npz_path, device, augment=True):
        print(f"📦 加载全尺寸物理数据并锁定至显存 [{device}]: {npz_path}")
        data = np.load(npz_path)
        
        self.device = device
        self.augment = augment
        
        # 直接载入显存 [N, 4, 145, 689]
        self.fields = torch.tensor(data['x'], dtype=torch.float32, device=device)
        self.labels = torch.tensor(data['y'], dtype=torch.float32, device=device)
        
        grid_x = torch.tensor(data['grid_x'], dtype=torch.float32, device=device) # [145, 689]
        grid_y = torch.tensor(data['grid_y'], dtype=torch.float32, device=device) # [145, 689]
        
        self.grid_x = 2.0 * (grid_x - grid_x.min()) / (grid_x.max() - grid_x.min() + 1e-8) - 1.0
        self.grid_y = 2.0 * (grid_y - grid_y.min()) / (grid_y.max() - grid_y.min() + 1e-8) - 1.0
        
        self.num_samples = len(self.fields)
        self.total_len = self.num_samples * 2 if augment else self.num_samples

    def get_batch(self, batch_size):
        indices = torch.randint(0, self.total_len, (batch_size,), device=self.device)
        
        base_indices = indices % self.num_samples
        is_flipped = (indices >= self.num_samples)
        
        flow = self.fields[base_indices].clone()
        grid_x = self.grid_x.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        grid_y = self.grid_y.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        cond = self.labels[base_indices].clone()
        
        if is_flipped.any():
            flip_mask = is_flipped
            flow[flip_mask] = torch.flip(flow[flip_mask], dims=[-2])
            flow[flip_mask, 2] = -flow[flip_mask, 2]
            grid_y[flip_mask] = -torch.flip(grid_y[flip_mask], dims=[-2])
            grid_x[flip_mask] = torch.flip(grid_x[flip_mask], dims=[-2])
            
        return flow, grid_x, grid_y, cond


# ==========================================
# 2. C-grid 物理空间微分算子
# ==========================================
def physical_spatial_gradient(f, x, y):
    if x.dim() == 3: x = x.unsqueeze(1)
    if y.dim() == 3: y = y.unsqueeze(1)

    df_dxi = F.pad(f[:, :, :, 1:] - f[:, :, :, :-1], (0, 1, 0, 0))
    df_deta = F.pad(f[:, :, 1:, :] - f[:, :, :-1, :], (0, 0, 0, 1))

    dx_dxi = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))
    dx_deta = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1))
    dy_dxi = F.pad(y[:, :, :, 1:] - y[:, :, :, :-1], (0, 1, 0, 0))
    dy_deta = F.pad(y[:, :, 1:, :] - y[:, :, :-1, :], (0, 0, 0, 1))

    J = dx_dxi * dy_deta - dx_deta * dy_dxi
    J_safe = torch.where(J.abs() < 1e-7, torch.ones_like(J) * 1e-7, J)

    df_dx = (df_dxi * dy_deta - df_deta * dy_dxi) / J_safe
    df_dy = (df_deta * dx_dxi - df_dxi * dx_deta) / J_safe

    return df_dx, df_dy, J_safe.abs()


def compute_vorticity_cgrid(u, v, grid_x, grid_y):
    dv_dx, _, _ = physical_spatial_gradient(v, grid_x, grid_y)
    _, du_dy, _ = physical_spatial_gradient(u, grid_x, grid_y)
    return dv_dx - du_dy


def compute_divergence_cgrid(u, v, grid_x, grid_y):
    du_dx, _, _ = physical_spatial_gradient(u, grid_x, grid_y)
    _, dv_dy, _ = physical_spatial_gradient(v, grid_x, grid_y)
    return du_dx + dv_dy


# ==========================================
# 3. 优化后的 PINN 物理 Loss
# ==========================================
def physics_informed_loss_optimized(pred_noise, true_noise, x0_pred, x0_true, grid_x, grid_y, bl_weight_cached, lambda_grad=0.05, lambda_vort=0.4, lambda_div=0.2):
    mse_noise_loss = F.mse_loss(pred_noise, true_noise)
    
    if bl_weight_cached.shape[2] != x0_pred.shape[2]:
        H_curr = x0_pred.shape[2]
        J_indices = torch.arange(H_curr, dtype=torch.float32, device=x0_pred.device)
        bl_decay = 9.0 * torch.exp(-J_indices / 6.0) + 1.0 
        bl_weight_cached = bl_decay.view(1, 1, H_curr, 1)

    loss_rho = F.l1_loss(x0_pred[:, 0:1] * bl_weight_cached, x0_true[:, 0:1] * bl_weight_cached)
    loss_u   = F.l1_loss(x0_pred[:, 1:2] * bl_weight_cached, x0_true[:, 1:2] * bl_weight_cached)
    loss_v   = F.l1_loss(x0_pred[:, 2:3] * bl_weight_cached, x0_true[:, 2:3] * bl_weight_cached) * 6.0 
    loss_p   = F.l1_loss(x0_pred[:, 3:4] * bl_weight_cached, x0_true[:, 3:4] * bl_weight_cached) * 2.5 
    
    direct_field_loss = loss_rho + loss_u + loss_v + loss_p

    pred_gx, pred_gy, Jacobian = physical_spatial_gradient(x0_pred, grid_x, grid_y)
    true_gx, true_gy, _ = physical_spatial_gradient(x0_true, grid_x, grid_y)
    
    grad_loss = F.mse_loss(pred_gx, true_gx) + F.mse_loss(pred_gy, true_gy)

    vort_pred = compute_vorticity_cgrid(x0_pred[:, 1:2], x0_pred[:, 2:3], grid_x, grid_y)
    vort_true = compute_vorticity_cgrid(x0_true[:, 1:2], x0_true[:, 2:3], grid_x, grid_y)
    vorticity_loss = F.l1_loss(vort_pred, vort_true)

    div_pred = compute_divergence_cgrid(x0_pred[:, 1:2], x0_pred[:, 2:3], grid_x, grid_y)
    div_true = compute_divergence_cgrid(x0_true[:, 1:2], x0_true[:, 2:3], grid_x, grid_y)
    divergence_loss = F.l1_loss(div_pred, div_true)

    return mse_noise_loss + direct_field_loss + lambda_grad * grad_loss + \
           lambda_vort * vorticity_loss + lambda_div * divergence_loss

def add_noise_fast(x0, t, sqrt_ab_table, sqrt_1_minus_ab_table):
    noise = torch.randn_like(x0)
    x_t = sqrt_ab_table[t] * x0 + sqrt_1_minus_ab_table[t] * noise
    return x_t, noise


# ==========================================
# 4. 主训练流程
# ==========================================
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else "."
    train_npz_path = os.path.abspath(os.path.join(current_dir, "../Results/Diffusion_airfoil_unsteady_full_train.npz"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动 145x689 全尺寸 Diffusion 极速训练引擎: {device}")
    
    torch.backends.cudnn.benchmark = True
    
    dataset = GPUDirectDataset(train_npz_path, device=device, augment=True)
    
    model = DiffusionUNet(flow_ch=4, coord_ch=2, cond_dim=128, base_ch=48).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
    
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    
    num_timesteps = 1000
    epochs = 1000 
    batch_size = 4
    steps_per_epoch = dataset.total_len // batch_size
    
    beta = torch.linspace(1e-4, 0.02, num_timesteps).to(device)
    alpha_bar = torch.cumprod(1.0 - beta, dim=0)
    sqrt_ab_table = torch.sqrt(alpha_bar).view(-1, 1, 1, 1)
    sqrt_1_minus_ab_table = torch.sqrt(1.0 - alpha_bar + 1e-8).view(-1, 1, 1, 1)
    
    H_size = dataset.fields.shape[2]  # 自动获取数据集真实 H 维度 (145)
    J_indices = torch.arange(H_size, dtype=torch.float32, device=device)
    bl_decay = 9.0 * torch.exp(-J_indices / 6.0) + 1.0 
    bl_weight_cached = bl_decay.view(1, 1, H_size, 1)

    global_pbar = tqdm(range(epochs), desc="🚀 145x689 全尺寸训练中")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    for epoch in global_pbar:
        model.train()
        epoch_loss = 0.0
        
        for _ in range(steps_per_epoch):
            flow, grid_x, grid_y, cond = dataset.get_batch(batch_size)
            
            t = torch.randint(0, num_timesteps, (batch_size,), device=device).long()
            x_t, true_noise = add_noise_fast(flow, t, sqrt_ab_table, sqrt_1_minus_ab_table)
            
            cond_input = cond.clone()
            drop_mask = (torch.rand((batch_size, 1), device=device) < 0.15).float()
            cond_input = cond_input * (1.0 - drop_mask)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                x0_pred = model(x_t, grid_x, grid_y, t, cond_input)
                pred_noise = (x_t - sqrt_ab_table[t] * x0_pred) / sqrt_1_minus_ab_table[t]
                
                loss = physics_informed_loss_optimized(
                    pred_noise, true_noise, x0_pred, flow, 
                    grid_x, grid_y, bl_weight_cached,
                    lambda_grad=0.05, lambda_vort=0.4, lambda_div=0.2
                )
            
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / steps_per_epoch
        scheduler.step()
        
        global_pbar.set_postfix({"Epoch": epoch + 1, "Loss": f"{avg_loss:.5f}", "LR": f"{optimizer.param_groups[0]['lr']:.2e}"})
        
        if (epoch + 1) % 500 == 0:
            save_path = os.path.join(current_dir, f"../Results/airfoil_diffusion_cgrid_full_ep{epoch+1}.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

    final_path = os.path.join(current_dir, "../Results/airfoil_diffusion_cgrid_full_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"✅ 全尺寸训练完成，权重已保存至: {final_path}")