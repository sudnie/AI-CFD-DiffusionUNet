#usr/bin/python3
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 🌟 1. 고해상도 적응 정렬 네트워크 도입
from model_utils import HighResDiffusionUNet

# ==========================================
# 1. VRAM 에 직접 고해상도 데이터셋 마운트
# ==========================================
class AirfoilHighResVRAMDataset(Dataset):
    def __init__(self, npz_path, device):
        print(f"📦 고품질 무확대 데이터 코드를 VRAM 에 로드 중 [{device}]: {npz_path}")
        data = np.load(npz_path)

        # fields 형태: [N, 4, 60, 301]
        self.fields = torch.tensor(data['x'], dtype=torch.float32, device=device)
        self.labels = torch.tensor(data['y'], dtype=torch.float32, device=device)

        # 원본 대형 격자 좌표 [60, 301] 의 실제 무차원 척도를 유지, 외야 제어 반경을 결코 파괴하지 않음
        self.grid_x = torch.tensor(data['grid_x'], dtype=torch.float32, device=device)
        self.grid_y = torch.tensor(data['grid_y'], dtype=torch.float32, device=device)

    def __len__(self):
        return len(self.fields)

    def __getitem__(self, idx):
        return self.fields[idx], self.grid_x, self.grid_y, self.labels[idx]

# ==========================================
# 2. 물리 유도 손실 및 초고속 노이즈 전략
# ==========================================
def spatial_gradient(tensor):
    """1 차 공간 유한차분: (60, 301) 의 실제 대형 크기에서 경계층 전단응력 기울기를 더 정확하게 포착"""
    grad_x = F.pad(tensor[:, :, :, 1:] - tensor[:, :, :, :-1], (0, 1, 0, 0))
    grad_y = F.pad(tensor[:, :, 1:, :] - tensor[:, :, :-1, :], (0, 0, 0, 1))
    return grad_x, grad_y

def physics_informed_loss_scheme_b(pred_noise, true_noise, x0_pred, x0_true, grid_x, grid_y, lambda_grad=1.0):
    """
    고품질 위상 적응 물리 손실 함수 - 구조화된 접안 경계층 대폭 가중 도입
    """
    # 1. 기본 노이즈 확산 손실
    mse_noise_loss = F.mse_loss(pred_noise, true_noise)

    """
    # 2. 공간 1 계도함수 기울기 제약 (근접장 가우스 물리 방어막과 함께)
    radius_sq = grid_x**2 + grid_y**2
    if len(radius_sq.shape) == 3:
        spatial_weight = torch.exp(-2.0 * radius_sq).unsqueeze(1)
    else:
        spatial_weight = torch.exp(-2.0 * radius_sq).unsqueeze(0).unsqueeze(0)
    spatial_weight = spatial_weight.expand_as(x0_pred)

    pred_grad_x, pred_grad_y = spatial_gradient(x0_pred)
    true_grad_x, true_grad_y = spatial_gradient(x0_true)

    grad_loss_matrix = F.mse_loss(pred_grad_x, true_grad_x, reduction='none') + \
                       F.mse_loss(pred_grad_y, true_grad_y, reduction='none')

    # 3. 🎯🎯🎯 핵심 신규: 구조화된 접안 경계층 물리 가중막 (Boundary Layer Boost) 🎯🎯🎯
    # 구조화된 격자에서 H 축 (크기 60) 은 J=0 이고 고체 벽면에서 멀어질수록 커짐
    # 우리는 H 축 인덱스 (J) 를 기반으로 벽면에서 바깥으로 급격히 감쇠하는 가중 계수를 생성합니다.
    # 가중 공식: weight = exp(-J / sigma), 여기서 sigma=3.0 은 경계층 두께 범위 제어 (~J=0~10 의 근접 벽면 급격 기울기 영역)
    H_size = x0_pred.shape[2]  # 60 이어야 함
    J_indices = torch.arange(H_size, dtype=torch.float32, device=x0_pred.device)# [60]

    # 경계층 감쇠 곡선: J=0 에서 가중 10.0, 바깥으로 급격히 감쇠하여 1.0 기준에 도달
    bl_decay = 9.0 * torch.exp(-J_indices / 3.0) + 1.0# [60] 형태

    # 이를 x0_pred 와 동일한 형태 [B, C, H, W] 로 브로드캐스트 확장
    bl_weight = bl_decay.view(1, 1, H_size, 1).expand_as(x0_pred)

    # 4. 기본 채널 공간 직접 제약 (경계층 가중 bl_weight 를 곱하여, 네트워크에 근접 벽면 정밀도를 강제로 요구)
    vel_direct_loss = F.l1_loss(x0_pred[:, 0:3] * bl_weight[:, 0:3], x0_true[:, 0:3] * bl_weight[:, 0:3])
    press_direct_loss = F.l1_loss(x0_pred[:, 3:4] * bl_weight[:, 3:4], x0_true[:, 3:4] * bl_weight[:, 3:4]) * 2.5

    # 경계층 가중을 기울기 손실에 동일하게 부여하여, 근접 벽면 법선 전단응력 (점성항) 이 강한 제약될 수 있도록 함
    grad_loss = (grad_loss_matrix * spatial_weight * bl_weight).mean()

    # 5. 원접경계 하드 제약 (Far-field Hard Constraint)
    if len(radius_sq.shape) == 3:
        far_field_mask = (radius_sq > 0.64).float().unsqueeze(1).expand_as(x0_pred)
    else:
        far_field_mask = (radius_sq > 0.64).float().unsqueeze(0).unsqueeze(0).expand_as(x0_pred)

    far_field_loss = F.l1_loss(x0_pred * far_field_mask, x0_true * far_field_mask) * 4.0

    # 6. 극값 강한 정렬 처벌
    max_penalty = F.mse_loss(x0_pred.max(dim=-1)[0].max(dim=-1)[0], x0_true.max(dim=-1)[0].max(dim=-1)[0])
    min_penalty = F.mse_loss(x0_pred.min(dim=-1)[0].min(dim=-1)[0], x0_true.min(dim=-1)[0].min(dim=-1)[0])
    extrema_loss = 0.5 * (max_penalty + min_penalty) """

    # 결합 총 손실
    return mse_noise_loss #+ vel_direct_loss + press_direct_loss + lambda_grad * grad_loss + extrema_loss + far_field_loss

# def physics_informed_loss_scheme_b(pred_noise, true_noise, x0_pred, x0_true, grid_x, grid_y, lambda_grad=1.0):
#     """
#     (60, 301) 물리면 최적화된 5 중 손실 청산 함수
#     """
#     # 1. 기본 노이즈 제거 손실
#     mse_noise_loss = F.mse_loss(pred_noise, true_noise)

#     # 2. 공간 1 계도함수 기울기 제약 (근접장 가우스 물리 방어막과 함께)
#     radius_sq = grid_x**2 + grid_y**2

#     # 🎯 견고성 재구성: 적응적 정렬 고차원 직사각형 행렬의 Batch 브로드캐스트 크기
#     if len(radius_sq.shape) == 3:
#         spatial_weight = torch.exp(-2.0 * radius_sq).unsqueeze(1)# [B, 1, 60, 301]
#     else:
#         # 단일 이미지거나 DataLoader 축소인 경우, 동적으로 Batch 및 Channel 축 확장
#         spatial_weight = torch.exp(-2.0 * radius_sq).unsqueeze(0).unsqueeze(0)# [1, 1, 60, 301]

#     spatial_weight = spatial_weight.expand_as(x0_pred)

#     pred_grad_x, pred_grad_y = spatial_gradient(x0_pred)
#     true_grad_x, true_grad_y = spatial_gradient(x0_true)

#     grad_loss = (F.mse_loss(pred_grad_x, true_grad_x, reduction='none') + \
#                  F.mse_loss(pred_grad_y, true_grad_y, reduction='none')) * spatial_weight

#     # 3. 기본 채널 공간 직접 제약
#     vel_direct_loss = F.l1_loss(x0_pred[:, 0:3], x0_true[:, 0:3])
#     press_direct_loss = F.l1_loss(x0_pred[:, 3:4], x0_true[:, 3:4]) * 2.5

#     # 4. 원접경계 하드 제약 (Far-field Hard Constraint)
#     if len(radius_sq.shape) == 3:
#         far_field_mask = (radius_sq > 0.64).float().unsqueeze(1).expand_as(x0_pred)
#     else:
#         far_field_mask = (radius_sq > 0.64).float().unsqueeze(0).unsqueeze(0).expand_as(x0_pred)

#     far_field_loss = F.l1_loss(x0_pred * far_field_mask, x0_true * far_field_mask) * 4.0

#     # 5. 극값 강한 정렬 처벌
#     max_penalty = F.mse_loss(x0_pred.max(dim=-1)[0].max(dim=-1)[0], x0_true.max(dim=-1)[0].max(dim=-1)[0])
#     min_penalty = F.mse_loss(x0_pred.min(dim=-1)[0].min(dim=-1)[0], x0_true.min(dim=-1)[0].min(dim=-1)[0])
#     extrema_loss = 0.5 * (max_penalty + min_penalty)

#     return mse_noise_loss + vel_direct_loss + press_direct_loss + lambda_grad * grad_loss.mean() + extrema_loss + far_field_loss

def add_noise_fast(x0, t, sqrt_ab_table, sqrt_1_minus_ab_table):
    noise = torch.randn_like(x0)
    x_t = sqrt_ab_table[t] * x0 + sqrt_1_minus_ab_table[t] * noise
    return x_t, noise

# ==========================================
# 3. 훈련 메인 엔진
# ==========================================
if __name__ == "__main__":
    import time

    # ==========================================
    # ⏱️ 물리 계산 버스: 고정밀 타이머 시작
    # ==========================================
    start_wall_time = time.time()
    if torch.cuda.is_available():
        # 🎯 핵심 학술 규범: GPU 비동기 아키텍처에서는 명시적인 동기화 차단이 필수, 실제 하드웨어 체산 시간 획득
        torch.cuda.synchronize()
    start_cuda_time = time.time()

    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else "."
    train_npz_path = os.path.abspath(os.path.join(current_dir, "../Results/HighRes_airfoil_train.npz"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 (60x301) 고품질 변 크기 물리 강화 훈련 엔진 시작: {device}")

    dataset = AirfoilHighResVRAMDataset(train_npz_path, device=device)
    dataloader = DataLoader(
        dataset,
        batch_size=16,          # 고해상도 VRAM 점유 큼, 16-32 미세 배치 파이프라인 권
        shuffle=True,
        num_workers=0,          # 메인 스레드 상시 직접 공급
        drop_last=False
    )

    model = HighResDiffusionUNet().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda')

    num_timesteps = 1000
    epochs = 5000

    # 1000 단계 확산 상수 표 생성
    beta = torch.linspace(1e-4, 0.02, num_timesteps).to(device)
    alpha_bar = torch.cumprod(1.0 - beta, dim=0)

    sqrt_ab_table = torch.sqrt(alpha_bar).view(-1, 1, 1, 1)
    sqrt_1_minus_ab_table = torch.sqrt(1.0 - alpha_bar + 1e-8).view(-1, 1, 1, 1)

    loss_history = []
    print("\n⚡ 물리 엔진 버스 연결 성공, 본격적인 행간 유동장 추적 청산 시작...")
    print("-" * 75)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for flow, grid_x, grid_y, cond in dataloader:
            t = torch.randint(0, num_timesteps, (flow.shape[0],), device=device).long()

            #无损加噪
            x_t, true_noise = add_noise_fast(flow, t, sqrt_ab_table, sqrt_1_minus_ab_table)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                # 예측 (60, 301) 원본 무노이즈 유동장
                x0_pred = model(x_t, grid_x, grid_y, t, cond)
                # 숨김 역노이즈
                pred_noise = (x_t - sqrt_ab_table[t] * x0_pred) / sqrt_1_minus_ab_table[t]

                loss = physics_informed_loss_scheme_b(
                    pred_noise, true_noise, x0_pred, flow, grid_x, grid_y, lambda_grad=1.0
                )

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # 🎯 핵심 수정: tqdm 제거, 표준적이고 깔끔한 행간 Epoch 정보 추적 인쇄 구현
        print(f"▶ Epoch [{epoch+1:04d}/{epochs}] | HighRes_PI_Loss: {avg_loss:.6f} | LearningRate: {current_lr:.2e}")

        # 가중치 안전 고착
        if (epoch + 1) % 500 == 0 or (epoch + 1) == epochs:
            save_path = os.path.join(current_dir, f"../Results/airfoil_diffusion_highres_ep{epoch+1}.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"   💾 [Checkpoint] 공기역학적 특성 가중치가 심층 잠금됨: {save_path}")

    print("-" * 75)
    # ==========================================
    # ⏱️ 물리 계산 버스: 고정밀 타이머 정산
    # ==========================================
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()

    total_wall_seconds = end_time - start_wall_time
    total_cuda_seconds = end_time - start_cuda_time

    # 초를 더 직관적인 시:분:초 형식으로 포매팅
    def format_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:.2f}"

    print("\n⏱️  ========================================================")
    print(f"📊 [HighRes-Diffusion 컴퓨팅 총비용 청산 보고서]")
    print(f"    -> 순수 네트워크 훈련 실행 시간: {format_time(total_cuda_seconds)}")
    print(f"    -> 전체 프로세스 IO 포함 총소요시간: {format_time(total_wall_seconds)}")
    print("============================================================\n")
    # 고품질 수렴 Lossграда الصور导出
    import matplotlib.pyplot as plt
    print("\n📈 고품질 훈련 Loss 수렴 스펙트럼 도표 내보내기 중...")
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, label='Train Loss', color='crimson', linewidth=2)
    plt.title('High-Res DiffusionUNet (60x301) 수렴 추적', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs', fontsize=11)
    plt.ylabel('PI-Loss Value', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    curve_save_path = os.path.abspath(os.path.join(current_dir, "../Results/loss_highres_convergence.png"))
    plt.savefig(curve_save_path, dpi=300, bbox_inches='tight')
    print(f"🎉 고해상도 수렴 곡선 고정 완료: {curve_save_path}")
    plt.show()

