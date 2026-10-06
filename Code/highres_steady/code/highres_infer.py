#!/usr/bin/python3
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 🌟 적응형 고품질 직사각형 노이즈 제거 네트워크
from model_utils import HighResDiffusionUNet

# ==========================================
# 🌟 x0 예측 기반 정통 역방향 샘플러 (고품질 변 크기 적응)
# ==========================================
@torch.no_grad()
def sample_flow_field_scheme_b(model, device, grid_x, grid_y, target_cond, timesteps=1000):
    model.eval()

    # 훈련時的 선형 스케줄러와 엄격 정렬
    beta = torch.linspace(1e-4, 0.02, timesteps).to(device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    # 🎯 동적 적응 현재 입력 대형 격자 H, W, 원본 고해상도 공간 안에서 직접 가우즈 노이즈 주입
    h, w = grid_x.shape[1], grid_x.shape[2]
    x_t = torch.randn((1, 4, h, w), device=device)

    pbar = tqdm(reversed(range(timesteps)), desc="🌪️方案 B ($x_0$예측) 역방향 노이즈 제거 진화 실행 중", total=timesteps)
    for t_idx in pbar:
        t = torch.full((1,), t_idx, device=device, dtype=torch.long)

        # 1. 이제 모델에서 배출된 것은 예측된 무노이즈 깔끔 유동장 x0_pred
        x0_pred = model(x_t, grid_x, grid_y, t, target_cond)

        # 2. 현재 단계의 스케줄러 상수 추출
        ab_t = alpha_bar[t_idx]

        if t_idx > 0:
            # 3. t > 0 일 때, 표준 x0 예측 공식을 통해 x_{t-1} 의 평균 부분 유도
            ab_t_prev = alpha_bar[t_idx - 1]

            # 무노이즈 예측 항과 현재 노이즈 보존 항의 가중계수 계산
            weight_x0 = torch.sqrt(ab_t_prev) * beta[t_idx] / (1.0 - ab_t)
            weight_xt = torch.sqrt(alpha[t_idx]) * (1.0 - ab_t_prev) / (1.0 - ab_t)

            mean = weight_x0 * x0_pred + weight_xt * x_t

            # 사후분산 무작위항 도입
            var = (1.0 - ab_t_prev) / (1.0 - ab_t) * beta[t_idx]
            noise = torch.randn_like(x_t)

            x_t = mean + torch.sqrt(var) * noise
        else:
            # 4. 최종 단계 (t=0)
            x_t = x0_pred

    return x_t

# ==========================================
# 🌟 CFDLib 와 절대 동일 단일 채널 상대오차 연산자
# ==========================================
def relative_error_calc(pred, true, eps=1e-8):
    """
    평탄화 후 단일 채널의 1 차 이산 상대오차 계산
    """
    return np.sum(np.abs(pred - true)) / (np.sum(np.abs(true)) + eps)

# ==========================================
# 2. 자동 검색 및 평가 메인 프로그램
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 고품질 변 크기方案 B 전용 유동장 평가 엔진 시작: {device}")

    # 🌟[테스트 사례 선택] 보고 싶은 결과工况 입력
    TARGET_MACH = 0.475
    TARGET_AOA = 6

    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else "."
    # 🎯 고품질 전용 가중치 파일 및 참조 데이터셋 경로 정렬
    weights_path = os.path.abspath(os.path.join(current_dir, "../Results/airfoil_diffusion_highres_ep5000.pth"))
    norm_path = os.path.abspath(os.path.join(current_dir, "../Results/normalization_factors_highres.npz"))

    train_data_path = os.path.abspath(os.path.join(current_dir, "../Results/HighRes_airfoil_train.npz"))
    test_data_path = os.path.abspath(os.path.join(current_dir, "../Results/HighRes_airfoil_test.npz"))

    # 🎯 적응적 고정밀 아키텍처 네트워크 도입
    model = HighResDiffusionUNet().to(device)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("✅ 고품질 물리 model 가중치 로드 완료.")
    else:
        raise FileNotFoundError(f"❌ model 가중치 파일을 찾을 수 없음: {weights_path}, 먼저 model 이 성공적으로 훈련되었음을 확인!")

    train_data = np.load(train_data_path)
    test_data = np.load(test_data_path)
    norm_factors = np.load(norm_path)

    f_min, f_max = norm_factors['fields_min'], norm_factors['fields_max']
    l_min, l_max = norm_factors['label_min'], norm_factors['label_max']
    grid_x_raw = train_data['grid_x']
    grid_y_raw = train_data['grid_y']

    # 전역 최근접 이웃 검색
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
    print(f"🔏 실제 유동장 소스 매칭 성공: 【{data_source}】 #내부 행 인덱스 {idx}")
    print(f"📊 [물리 정렬 검증] 최종 인출된 공기역학적工况:")
    print(f"    -> 실제 세계 맥수 Mach: {cond_phys[0]:.4f} (기대: {TARGET_MACH:.4f})")
    print(f"    -> 실제 세계 공격각 AoA:    {cond_phys[1]:.4f}° (기대: {TARGET_AOA:.1f}°)")

    # 형식 입력 특징
    cond_tensor = torch.tensor(target_norm, dtype=torch.float32, device=device).unsqueeze(0)
    grid_x_tensor = torch.tensor(grid_x_raw, dtype=torch.float32, device=device).unsqueeze(0)
    grid_y_tensor = torch.tensor(grid_y_raw, dtype=torch.float32, device=device).unsqueeze(0)

    # ⏱️ 고정밀 타이머 시작
    start_wall_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_cuda_time = time.time()

    # 🎯 고정된 img_size 제한 제거, 내부 연산자가 대형 격자 생성에 적응
    pred_field_norm = sample_flow_field_scheme_b(model, device, grid_x_tensor, grid_y_tensor, cond_tensor)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_time = time.time() - start_cuda_time
    print(f"⏱️ 샘플 계산 소요시간: {elapsed_time:.2f} 초")

    # 반정규화
    def denormalize(field_tensor):
        arr = field_tensor.cpu().numpy()
        if len(arr.shape) == 4: arr = arr[0]
        phys_0_1 = (arr + 1.0) / 2.0
        f_min_c = f_min[0, :, 0, 0].reshape(4, 1, 1)
        f_max_c = f_max[0, :, 0, 0].reshape(4, 1, 1)
        return phys_0_1 * (f_max_c - f_min_c) + f_min_c

    gt_phys = denormalize(torch.tensor(gt_fields_norm))
    pred_phys = denormalize(pred_field_norm)
    err_phys = np.abs(gt_phys - pred_phys)

    # =================================================================
    # 🌟 핵심 융합 재구성구역: 이중 체계 오류 전청산 🌟
    # =================================================================
    eps = 1e-5
    ch_u, ch_v, ch_p = 1, 2, 3 # 1-U속도, 2-V속도, 3-Pressure압력

    # ----- 체계1: 전통적인 단일channel 이산 상대오차 (relative_error 대응) -----
    error_u_relative = relative_error_calc(pred_phys[ch_u].flatten(), gt_phys[ch_u].flatten(), eps)
    error_v_relative = relative_error_calc(pred_phys[ch_v].flatten(), gt_phys[ch_v].flatten(), eps)
    error_p_relative = relative_error_calc(pred_phys[ch_p].flatten(), gt_phys[ch_p].flatten(), eps)

    # ----- 체계2: 학계常用的 표준 channel 상대 L2 오차 -----
    l2_error_u = np.linalg.norm(gt_phys[ch_u] - pred_phys[ch_u]) / (np.linalg.norm(gt_phys[ch_u]) + eps)
    l2_error_p = np.linalg.norm(gt_phys[ch_p] - pred_phys[ch_p]) / (np.linalg.norm(gt_phys[ch_p]) + eps)

    # ----- 체계3: 【Table 4 핵심】전channel 융합 공간 상대 L2 오차 백분율 -----
    u_pred_f, u_true_f = pred_phys[ch_u].flatten(), gt_phys[ch_u].flatten()
    v_pred_f, v_true_f = pred_phys[ch_v].flatten(), gt_phys[ch_v].flatten()
    p_pred_f, p_true_f = pred_phys[ch_p].flatten(), gt_phys[ch_p].flatten()

    res_sq = np.sum((u_pred_f - u_true_f)**2) + np.sum((v_pred_f - v_true_f)**2) + np.sum((p_pred_f - p_true_f)**2)
    den_sq = np.sum(u_true_f**2) + np.sum(v_true_f**2) + np.sum(p_true_f**2)
    case_combined_l2_error = np.sqrt(res_sq / den_sq) * 100

    # 극값 최고오차 좌표점 위치
    max_err_idx_u = np.unravel_index(np.argmax(err_phys[ch_u]), err_phys[ch_u].shape)
    max_err_idx_p = np.unravel_index(np.argmax(err_phys[ch_p]), err_phys[ch_p].shape)
    max_err_x_u, max_err_y_u = grid_x_raw[max_err_idx_u], grid_y_raw[max_err_idx_u]
    max_err_x_p, max_err_y_p = grid_x_raw[max_err_idx_p], grid_y_raw[max_err_idx_p]

    # 콘솔 전청산 인쇄
    print("\n📈 ========================================================")
    print(f"📊 [Diffusion model 전공기역학적 channel 다차원 오류 보고서]")
    print(f"    >> 백업 단일channel 이산 상대오차 (relative_error 매핑):")
    print(f"       -> Error u: {error_u_relative:.6e}")
    print(f"       -> Error v: {error_v_relative:.6e}")
    print(f"       -> Error p: {error_p_relative:.6e}")
    print(f"    >> 단일channel 표준 상대 L2 오차:")
    print(f"       -> U-Speed 상대 L2 오차: {l2_error_u * 100:.3f}%")
    print(f"       -> Pressure   상대 L2 오차: {l2_error_p * 100:.3f}%")
    print(f"    -------------------------------------------------------")
    print(f"    >> 【Table 4 핵심】전channel 융합 공간 상대 L2 오차: {case_combined_l2_error:.4f}%")
    print("===========================================================\n")

    # ==========================================
    # 3. 정상류장 3x2 표준 가시화 출력 (연속색_updates)
    # ==========================================
    print("🎨 고품질 매끄러운 유동장 및 오류 포착 지도 렌더링 중...")
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f"High-Res Original Mesh 검증 행렬 (Table 4 L2: {case_combined_l2_error:.3f}%)\nMach={TARGET_MACH:.3f}, AoA={TARGET_AOA:.1f}°", fontsize=13, fontweight='bold')

    cmap, err_cmap = 'jet', 'magma'

    # 🌟 모든 contourf 를 pcolormesh(..., shading='gouraud')로 치환하여 연속 부드럽게 전환
    # --- 행 1: U 속도장 ---
    axes[0, 0].set_title(f"Ground Truth - U Velocity (Relative Err: {error_u_relative*100:.2f}%)")
    fig.colorbar(axes[0, 0].pcolormesh(grid_x_raw, grid_y_raw, gt_phys[ch_u], shading='gouraud', cmap=cmap), ax=axes[0, 0])

    axes[0, 1].set_title(f"Prediction - U Velocity (Channel $L_2$: {l2_error_u*100:.2f}%)")
    fig.colorbar(axes[0, 1].pcolormesh(grid_x_raw, grid_y_raw, pred_phys[ch_u], shading='gouraud', cmap=cmap), ax=axes[0, 1])

    # --- 행 2: P 압력장 ---
    axes[1, 0].set_title(f"Ground Truth - Pressure (Relative Err: {error_p_relative*100:.2f}%)")
    fig.colorbar(axes[1, 0].pcolormesh(grid_x_raw, grid_y_raw, gt_phys[ch_p], shading='gouraud', cmap=cmap), ax=axes[1, 0])

    axes[1, 1].set_title(f"Prediction - Pressure (Channel $L_2$: {l2_error_p*100:.2f}%)")
    fig.colorbar(axes[1, 1].pcolormesh(grid_x_raw, grid_y_raw, pred_phys[ch_p], shading='gouraud', cmap=cmap), ax=axes[1, 1])

    # --- 행 3: 절대오차장 + 红十字十字星으로 최대오차 최고점 표시 ---
    # U 속도 절대오차 서브도
    axes[2, 0].set_title("Absolute Error - U Velocity")
    im_err_u = axes[2, 0].pcolormesh(grid_x_raw, grid_y_raw, err_phys[ch_u], shading='gouraud', cmap=err_cmap)
    fig.colorbar(im_err_u, ax=axes[2, 0])
    axes[2, 0].scatter(max_err_x_u, max_err_y_u, color='red', marker='x', s=120, linewidths=2.5, label='Max Error Point', zorder=5)
    circle_u = plt.Circle((max_err_x_u, max_err_y_u), 0.08, color='red', fill=False, linewidth=1.5, linestyle='--', zorder=5)
    axes[2, 0].add_patch(circle_u)
    axes[2, 0].legend(loc='upper right', fontsize=8)

    # 압력 절대오차 서브도
    axes[2, 1].set_title("Absolute Error - Pressure")
    im_err_p = axes[2, 1].pcolormesh(grid_x_raw, grid_y_raw, err_phys[ch_p], shading='gouraud', cmap=err_cmap)
    fig.colorbar(im_err_p, ax=axes[2, 1])
    axes[2, 1].scatter(max_err_x_p, max_err_y_p, color='red', marker='x', s=120, linewidths=2.5, label='Max Error Point', zorder=5)
    circle_p = plt.Circle((max_err_x_p, max_err_y_p), 0.08, color='red', fill=False, linewidth=1.5, linestyle='--', zorder=5)
    axes[2, 1].add_patch(circle_p)
    axes[2, 1].legend(loc='upper right', fontsize=8)

    for ax in axes.flatten():
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()

