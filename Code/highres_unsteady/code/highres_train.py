#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
import torch

# ==========================================
# 1. 하드웨어 장치 감지 및 CUDA 강제 바인딩
# ==========================================
print("=" * 60)
print(f"Python 인터프리터 경로: {sys.executable}")
print(f"PyTorch 버전: {torch.__version__}")

cuda_available = torch.cuda.is_available()
print(f"CUDA 사용 가능: {cuda_available}")

if cuda_available:
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    device_count = torch.cuda.device_count()
    print(f"{device_count} 개의 GPU 그래픽카드 감지")
    print(f"🚀 그래픽카드 고정: {gpu_name}")
else:
    device = torch.device("cpu")
    print("⚠️ 경고: GPU 를 감지하지 못했습니다. CPU 모드로 격퇴합니다 (속도 느림)")

print(f"🚀 Diffusion 초고속 훈련 엔진 시작: {device}")
print("=" * 60)
sys.stdout.flush()

"""
임의 크기 (풀 크기 또는 다운샘플) 에 자동으로 적응하는 DiffusionUNet 훈련 스크립트
"""

import os
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# model 정의 가져오기 (model_utils 에 DiffusionUNet 존재 확인 필요)
from model_utils import DiffusionUNet

# ==========================================
# 2. 데이터셋 접미사 선택 ("full" 또는 "down4" 등)
# ==========================================
DATA_SUFFIX = "down1"  # "full" 등으로 변경 가능
TRAIN_NPZ = f"Diffusion_airfoil_unsteady_{DATA_SUFFIX}_train.npz"

# ==========================================
# 3. GPU 메모리 직접 배치 데이터 (: 자동 크기 적응)
# ==========================================
class GPUDirectDataset:
    def __init__(self, npz_path, device, augment=True):
        print(f"📦 물리 데이터 로드 및 VRAM 에 고정 중 [{device}]: {npz_path}")
        data = np.load(npz_path)

        self.device = device
        self.augment = augment

        # VRAM 에 직접 로드 [N, C, H, W]
        self.fields = torch.tensor(data['x'], dtype=torch.float32, device=device)
        self.labels = torch.tensor(data['y'], dtype=torch.float32, device=device)

        grid_x = torch.tensor(data['grid_x'], dtype=torch.float32, device=device)  # [H, W]
        grid_y = torch.tensor(data['grid_y'], dtype=torch.float32, device=device)  # [H, W]

        # 그리드 좌표 [-1, 1] 로 정규화 (모델 입력용)
        self.grid_x = 2.0 * (grid_x - grid_x.min()) / (grid_x.max() - grid_x.min() + 1e-8) - 1.0
        self.grid_y = 2.0 * (grid_y - grid_y.min()) / (grid_y.max() - grid_y.min() + 1e-8) - 1.0

        self.num_samples = len(self.fields)
        self.total_len = self.num_samples * 2 if augment else self.num_samples

        # 공간 크기 가져오기 (나중에 사용)
        self.H, self.W = self.fields.shape[2], self.fields.shape[3]
        print(f"📐 데이터 공간 크기: H={self.H}, W={self.W}")

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
            flow[flip_mask] = torch.flip(flow[flip_mask], dims=[-2])   # H 축을 따라 뒤집기
            flow[flip_mask, 2] = -flow[flip_mask, 2]                   # v 성분 부호 반전
            grid_y[flip_mask] = -torch.flip(grid_y[flip_mask], dims=[-2])
            grid_x[flip_mask] = torch.flip(grid_x[flip_mask], dims=[-2])

        return flow, grid_x, grid_y, cond

# ==========================================
# 4. C-그리드 물리 공간 미분 연산자 (크기 의존성 없음)
# ==========================================
def physical_spatial_gradient(f, x, y):
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if y.dim() == 3:
        y = y.unsqueeze(1)

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
# 5. 최적화된 PINN 물리 Loss (动态가변 경계층 가중치 사용)
# ==========================================
def physics_informed_loss_optimized(pred_noise, true_noise, x0_pred, x0_true,
                                    grid_x, grid_y, bl_weight_cached,
                                    lambda_grad=0.05, lambda_vort=0.4, lambda_div=0.2):
    mse_noise_loss = F.mse_loss(pred_noise, true_noise)

    # 경계층 가중치 동적 조정 (현재 배치 H 가 캐시와 불일치하면 다시 계산)
    H_curr = x0_pred.shape[2]
    if bl_weight_cached.shape[2] != H_curr:
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

    return mse_noise_loss + direct_field_loss + lambda_grad * grad_loss + lambda_vort * vorticity_loss + lambda_div * divergence_loss

def add_noise_fast(x0, t, sqrt_ab_table, sqrt_1_minus_ab_table):
    noise = torch.randn_like(x0)
    x_t = sqrt_ab_table[t] * x0 + sqrt_1_minus_ab_table[t] * noise
    return x_t, noise

# ==========================================
# 6. 메인 훈련 절차 (자동 크기 적응)
# ==========================================
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else "."

    # 데이터 파일 경로 생성
    train_npz_path = os.path.abspath(os.path.join(current_dir, f"../Results/{TRAIN_NPZ}"))
    if not os.path.exists(train_npz_path):
        raise FileNotFoundError(f"훈련 데이터 없음: {train_npz_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Diffusion 훈련 엔진 시작: {device}")

    torch.backends.cudnn.benchmark = True

    # 데이터셋 로드 (자동 H, W 획득)
    dataset = GPUDirectDataset(train_npz_path, device=device, augment=True)

    # 모델 초기화 (입력 채널: 4개 물리장 + 2개 그리드 좌표, 조건 차원 128, 기본 채널 48)
    model = DiffusionUNet(flow_ch=4, coord_ch=2, cond_dim=128, base_ch=48).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    num_timesteps = 1000
    epochs = 3000
    batch_size = 8
    steps_per_epoch = dataset.total_len // batch_size

    beta = torch.linspace(1e-4, 0.02, num_timesteps).to(device)
    alpha_bar = torch.cumprod(1.0 - beta, dim=0)
    sqrt_ab_table = torch.sqrt(alpha_bar).view(-1, 1, 1, 1)
    sqrt_1_minus_ab_table = torch.sqrt(1.0 - alpha_bar + 1e-8).view(-1, 1, 1, 1)

    # 경계층 가중치: 실제 H 에 따라 동적 계산 (초기 자리, loss 에서 다시 확인)
    H_size = dataset.H
    J_indices = torch.arange(H_size, dtype=torch.float32, device=device)
    bl_decay = 9.0 * torch.exp(-J_indices / 6.0) + 1.0
    bl_weight_cached = bl_decay.view(1, 1, H_size, 1)

    global_pbar = tqdm(range(epochs), desc=f"🚀 {DATA_SUFFIX} 훈련 중")
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
            save_path = os.path.join(current_dir, f"../Results/airfoil_diffusion_cgrid_{DATA_SUFFIX}_ep{epoch+1}.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

    final_path = os.path.join(current_dir, f"../Results/airfoil_diffusion_cgrid_{DATA_SUFFIX}_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"✅ 훈련 완료, 가중치 저장됨: {final_path}")

