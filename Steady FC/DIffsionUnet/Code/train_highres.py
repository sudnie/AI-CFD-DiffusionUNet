#/usr/bin/python3
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 🌟 1. 引入高保真、自适应对齐网络
from model_utils import HighResDiffusionUNet


# ==========================================
# 1. 显存直接挂载 HighRes Dataset
# ==========================================
class AirfoilHighResVRAMDataset(Dataset):

  def __init__(self, npz_path, device):
    print(f"📦 正在加载高保真未缩放数据至显存 [{device}]: {npz_path}")
    data = np.load(npz_path)

    # fields 形状：[N, 4, 60, 301]
    self.fields = torch.tensor(data["x"], dtype=torch.float32, device=device)
    self.labels = torch.tensor(data["y"], dtype=torch.float32, device=device)

    # 保持原始大网格坐标 [60, 301] 的真实无量纲尺度，绝不破坏外场控制半径
    self.grid_x = torch.tensor(
        data["grid_x"], dtype=torch.float32, device=device
    )
    self.grid_y = torch.tensor(
        data["grid_y"], dtype=torch.float32, device=device
    )

  def __len__(self):
    return len(self.fields)

  def __getitem__(self, idx):
    return self.fields[idx], self.grid_x, self.grid_y, self.labels[idx]


# ==========================================
# 2. 物理引导的损失与极速加噪策略
# ==========================================
def spatial_gradient(tensor):
  """一阶空间有限差分：在 (60, 301) 真实大尺寸下更精准地捕捉边界层切应力梯度"""
  grad_x = F.pad(tensor[:, :, :, 1:] - tensor[:, :, :, :-1], (0, 1, 0, 0))
  grad_y = F.pad(tensor[:, :, 1:, :] - tensor[:, :, :-1, :], (0, 0, 0, 1))
  return grad_x, grad_y


def physics_informed_loss_scheme_b(
    pred_noise,
    true_noise,
    x0_pred,
    x0_true,
    grid_x,
    grid_y,
    lambda_grad=1.0,
    use_wall_normal_mask=True,  # 🎯 新增控制开关标志位
):
  """高保真拓扑自适应物理损失函数 - 可选结构化贴体边界层大幅度加权"""
  # 1. 基础噪声扩散损失
  mse_noise_loss = F.mse_loss(pred_noise, true_noise)

  # 2. 空间一阶导数梯度约束（配合近场高斯物理防御罩）
  radius_sq = grid_x**2 + grid_y**2
  if len(radius_sq.shape) == 3:
    spatial_weight = torch.exp(-2.0 * radius_sq).unsqueeze(1)
  else:
    spatial_weight = torch.exp(-2.0 * radius_sq).unsqueeze(0).unsqueeze(0)
  spatial_weight = spatial_weight.expand_as(x0_pred)

  pred_grad_x, pred_grad_y = spatial_gradient(x0_pred)
  true_grad_x, true_grad_y = spatial_gradient(x0_true)

  grad_loss_matrix = F.mse_loss(
      pred_grad_x, true_grad_x, reduction="none"
  ) + F.mse_loss(pred_grad_y, true_grad_y, reduction="none")

  # 3. 🎯🎯🎯 结构化贴体边界层物理加权罩开关控制 🎯🎯🎯
  if use_wall_normal_mask:
    H_size = x0_pred.shape[2]  # H = 60
    J_indices = torch.arange(
        H_size, dtype=torch.float32, device=x0_pred.device
    )

    # 边界层衰减曲线：J=0时权重为10.0，向外迅速衰减至1.0基准
    bl_decay = 9.0 * torch.exp(-J_indices / 3.0) + 1.0  # 形状为 [60]
    bl_weight = bl_decay.view(1, 1, H_size, 1).expand_as(x0_pred)
  else:
    # 🌟 开启消融模式（OFF）：全空间赋予全 1.0 恒等权重（即关闭近壁面加权）
    bl_weight = torch.ones_like(x0_pred)

  # 4. 基础通道空间直接约束
  vel_direct_loss = F.l1_loss(
      x0_pred[:, 0:3] * bl_weight[:, 0:3], x0_true[:, 0:3] * bl_weight[:, 0:3]
  )
  press_direct_loss = (
      F.l1_loss(
          x0_pred[:, 3:4] * bl_weight[:, 3:4],
          x0_true[:, 3:4] * bl_weight[:, 3:4],
      )
      * 2.5
  )

  # 梯度损失加权
  grad_loss = (grad_loss_matrix * spatial_weight * bl_weight).mean()

  # 5. 远场边界硬约束（Far-field Hard Constraint）
  if len(radius_sq.shape) == 3:
    far_field_mask = (
        (radius_sq > 0.64).float().unsqueeze(1).expand_as(x0_pred)
    )
  else:
    far_field_mask = (
        (radius_sq > 0.64).float().unsqueeze(0).unsqueeze(0).expand_as(x0_pred)
    )

  far_field_loss = (
      F.l1_loss(x0_pred * far_field_mask, x0_true * far_field_mask) * 4.0
  )

  # 6. 极值强对齐惩罚
  max_penalty = F.mse_loss(
      x0_pred.max(dim=-1)[0].max(dim=-1)[0],
      x0_true.max(dim=-1)[0].max(dim=-1)[0],
  )
  min_penalty = F.mse_loss(
      x0_pred.min(dim=-1)[0].min(dim=-1)[0],
      x0_true.min(dim=-1)[0].min(dim=-1)[0],
  )
  extrema_loss = 0.5 * (max_penalty + min_penalty)

  # 结合总损失
  return (
      mse_noise_loss
      + vel_direct_loss
      + press_direct_loss
      + lambda_grad * grad_loss
      + extrema_loss
      + far_field_loss
  )


def add_noise_fast(x0, t, sqrt_ab_table, sqrt_1_minus_ab_table):
  noise = torch.randn_like(x0)
  x_t = sqrt_ab_table[t] * x0 + sqrt_1_minus_ab_table[t] * noise
  return x_t, noise


# ==========================================
# 3. 训练主引擎
# ==========================================
if __name__ == "__main__":
  # 🌟【全局掩膜开关配置】
  # True  : 开启 Wall-Normal Mask (正常训练模式)
  # False : 关闭 Wall-Normal Mask (消融实验模式, 用于生成对照组)
  USE_WALL_NORMAL_MASK = False

  start_wall_time = time.time()
  if torch.cuda.is_available():
    torch.cuda.synchronize()
  start_cuda_time = time.time()

  current_dir = (
      os.path.dirname(os.path.abspath(__file__))
      if "__file__" in locals()
      else "."
  )
  train_npz_path = os.path.abspath(
      os.path.join(current_dir, "../Results/HighRes_airfoil_train.npz")
  )

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  mask_status_str = "ON" if USE_WALL_NORMAL_MASK else "OFF (Ablation Mode)"
  print(
      f"🚀 启动 (60x301) 高保真变尺寸物理强化训练引擎: {device} | Wall-Normal Mask:"
      f" [{mask_status_str}]"
  )

  dataset = AirfoilHighResVRAMDataset(train_npz_path, device=device)
  dataloader = DataLoader(
      dataset, batch_size=16, shuffle=True, num_workers=0, drop_last=False
  )

  model = HighResDiffusionUNet().to(device)

  optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-5)
  scaler = torch.amp.GradScaler("cuda")

  num_timesteps = 1000
  epochs = 5000

  # 建立 1000 步扩散常数表
  beta = torch.linspace(1e-4, 0.02, num_timesteps).to(device)
  alpha_bar = torch.cumprod(1.0 - beta, dim=0)

  sqrt_ab_table = torch.sqrt(alpha_bar).view(-1, 1, 1, 1)
  sqrt_1_minus_ab_table = torch.sqrt(1.0 - alpha_bar + 1e-8).view(-1, 1, 1, 1)

  loss_history = []
  print("\n⚡ 物理引擎总线连接成功，正式启动逐行流场跟踪清算...")
  print("-" * 75)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=epochs, eta_min=1e-6
  )

  for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for flow, grid_x, grid_y, cond in dataloader:
      t = torch.randint(
          0, num_timesteps, (flow.shape[0],), device=device
      ).long()

      x_t, true_noise = add_noise_fast(
          flow, t, sqrt_ab_table, sqrt_1_minus_ab_table
      )

      optimizer.zero_grad()
      with torch.amp.autocast("cuda"):
        x0_pred = model(x_t, grid_x, grid_y, t, cond)
        pred_noise = (
            x_t - sqrt_ab_table[t] * x0_pred
        ) / sqrt_1_minus_ab_table[t]

        # 🎯 传入开关标志位变量 USE_WALL_NORMAL_MASK
        loss = physics_informed_loss_scheme_b(
            pred_noise,
            true_noise,
            x0_pred,
            flow,
            grid_x,
            grid_y,
            lambda_grad=1.0,
            use_wall_normal_mask=USE_WALL_NORMAL_MASK,
        )

      scaler.scale(loss).backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      scaler.step(optimizer)
      scaler.update()

      epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    loss_history.append(avg_loss)

    current_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()

    print(
        f"▶ Epoch [{epoch+1:04d}/{epochs}] | HighRes_PI_Loss: {avg_loss:.6f} |"
        f" LearningRate: {current_lr:.2e}"
    )

    # 权重安全固化（若为消融模式，保存在带有 ablation 后缀的文件中）
    if (epoch + 1) % 500 == 0 or (epoch + 1) == epochs:
      suffix = (
          "highres" if USE_WALL_NORMAL_MASK else "highres_ablation_nomask"
      )
      save_path = os.path.join(
          current_dir, f"../Results/airfoil_diffusion_{suffix}_ep{epoch+1}.pth"
      )
      os.makedirs(os.path.dirname(save_path), exist_ok=True)
      torch.save(model.state_dict(), save_path)
      print(f"   💾 [Checkpoint] 气动特征权重已深度锁存至: {save_path}")

  print("-" * 75)
  if torch.cuda.is_available():
    torch.cuda.synchronize()
  end_time = time.time()

  total_wall_seconds = end_time - start_wall_time
  total_cuda_seconds = end_time - start_cuda_time

  def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:.2f}"

  print("\n⏱️  ========================================================")
  print("📊 [HighRes-Diffusion 算力总开销清算报告]")
  print(f"    -> 纯网络训练执行用时 : {format_time(total_cuda_seconds)}")
  print(f"    -> 全流程含IO总吞吐用时: {format_time(total_wall_seconds)}")
  print("============================================================\n")

  import matplotlib.pyplot as plt

  print("\n📈 正在导出高保真训练 Loss 收敛谱图...")
  plt.figure(figsize=(10, 5))
  plt.plot(
      range(1, len(loss_history) + 1),
      loss_history,
      label="Train Loss",
      color="crimson",
      linewidth=2,
  )
  plt.title(
      f'High-Res DiffusionUNet (60x301) Convergence Track'
      f' (Mask={mask_status_str})',
      fontsize=12,
      fontweight="bold",
  )
  plt.xlabel("Epochs", fontsize=11)
  plt.ylabel("PI-Loss Value", fontsize=11)
  plt.grid(True, linestyle="--", alpha=0.5)
  plt.legend()

  suffix = "highres" if USE_WALL_NORMAL_MASK else "highres_ablation_nomask"
  curve_save_path = os.path.abspath(
      os.path.join(current_dir, f"../Results/loss_{suffix}_convergence.png")
  )
  plt.savefig(curve_save_path, dpi=300, bbox_inches="tight")
  print(f"🎉 高清收敛曲线固化完毕: {curve_save_path}")
  plt.show()