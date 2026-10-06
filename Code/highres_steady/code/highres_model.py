#usr/bin/python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

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

class ConditionalResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.adain1 = AdaIN(cond_dim, out_channels)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.adain2 = AdaIN(cond_dim, out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        h = self.adain1(self.conv1(x), cond)
        h = self.adain2(self.conv2(self.act(h)), cond)
        return h + self.shortcut(x)

class HighResDiffusionUNet(nn.Module):
    """
    적응형 고해상도 노이즈 제거 네트워크 4 단 위상 아키텍처로 재구성 (60 301) 임의 원본 크기 입력 완벽 호환
    """
    def __init__(self, flow_ch=4, coord_ch=2, cond_dim=128):
        super().__init__()
        self.init_conv = nn.Conv2d(flow_ch + coord_ch, 64, 3, padding=1)
        
        self.time_mlp = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.physics_mlp = nn.Sequential(nn.Linear(2, cond_dim // 2), nn.SiLU(), nn.Linear(cond_dim // 2, cond_dim))

        # 🎯 핵심 재구성 종단 위상 경로 4 단으로 업그레이드 채널 단계별 배로 확장 심층 다양체 표현
        # 다운샘플링 경로 (Encoder)
        self.down1 = ConditionalResBlock(64, 64, cond_dim)
        self.down2 = ConditionalResBlock(64, 128, cond_dim)
        self.down3 = ConditionalResBlock(128, 256, cond_dim)
        self.down4 = ConditionalResBlock(256, 512, cond_dim)  # 추가 최하단 물리 병목층 (Bottleneck)
        
        # 업 샘플 경로 (Decoder)
        self.up1 = ConditionalResBlock(512 + 256, 256, cond_dim)  # 추가 대응하는 제 1 층 디코딩
        self.up2 = ConditionalResBlock(256 + 128, 128, cond_dim)
        self.up3 = ConditionalResBlock(128 + 64, 64, cond_dim)
        self.up4 = ConditionalResBlock(64 + 64, 64, cond_dim)
        
        self.final_conv = nn.Conv2d(64, flow_ch, 1)

    def forward(self, x_t, grid_x, grid_y, t, physics_cond):
        # 1. 동적 공간 격자 물리 차원 정렬
        if len(grid_x.shape) == 2:
            grid_x = grid_x.unsqueeze(0).expand(x_t.shape[0], -1, -1)
            grid_y = grid_y.unsqueeze(0).expand(x_t.shape[0], -1, -1)
        if len(grid_x.shape) == 3:
            grid_x = grid_x.unsqueeze(1)
            grid_y = grid_y.unsqueeze(1)

        # 공간 인식 경로 활성화: 채널 하드 병합 [B, 6, H, W]
        x = torch.cat([x_t, grid_x, grid_y], dim=1)
        orig_h, orig_w = x.shape[2], x.shape[3]

        # 2. ⚡ 기하학적 공간 패딩 업그레이드 (4 층 네트워크는 2^3=8 로 나누어 떨어져야 함, 비대칭ม้า奇수 절단 방지)
        # 대상 (60, 301) -> H 에 4 픽셀 패딩으로 64 로, W 에 3 픽셀 패딩으로 304 로
        pad_h = (8 - orig_h % 8) % 8
        pad_w = (8 - orig_w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

        # 3. 외부 거시 제어 조건 임베딩
        cond = self.time_mlp(get_timestep_embedding(t, 128)) + self.physics_mlp(physics_cond)
        
        # 4. 순방향 심층 특징 인코딩 (4-Layer Encoder)
        x0 = self.init_conv(x)
        d1 = self.down1(x0, cond)
        
        d2_in = F.max_pool2d(d1, 2)
        d2 = self.down2(d2_in, cond)
        
        d3_in = F.max_pool2d(d2, 2)
        d3 = self.down3(d3_in, cond)
        
        d4_in = F.max_pool2d(d3, 2)  # 다시 다운샘플링
        d4 = self.down4(d4_in, cond) # 가장 깊은 층 특징 공간으로

        # 5. 역방향 적응적 크기 정렬 디코딩 (4-Layer Decoder + Skip Connection)
        # 명시적으로 동적 이전 층의 실제 입력 높이/너비를 고정, 모든 단일 픽셀 오류 방지
        u1_up = F.interpolate(d4, size=(d3.shape[2], d3.shape[3]), mode='nearest')
        u1 = self.up1(torch.cat([u1_up, d3], dim=1), cond)
        
        u2_up = F.interpolate(u1, size=(d2.shape[2], d2.shape[3]), mode='nearest')
        u2 = self.up2(torch.cat([u2_up, d2], dim=1), cond)
        
        u3_up = F.interpolate(u2, size=(d1.shape[2], d1.shape[3]), mode='nearest')
        u3 = self.up3(torch.cat([u3_up, d1], dim=1), cond)
        
        u4 = self.up4(torch.cat([u3, x0], dim=1), cond)
        out = self.final_conv(u4)
        
        # 6. ⚡ 물리 특징 역류 절단:剛才 동적으로 패딩한 픽셀을 제거, 밀리초 단위 무손실 원본 크기로 복원
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :orig_h, :orig_w]
        return out


