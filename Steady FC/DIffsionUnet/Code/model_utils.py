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
    自适应高保真去噪网络：重构为 4 层拓扑架构，完美契合 (60, 301) 任意未缩放输入
    """
    def __init__(self, flow_ch=4, coord_ch=2, cond_dim=128):
        super().__init__()
        self.init_conv = nn.Conv2d(flow_ch + coord_ch, 64, 3, padding=1)
        
        self.time_mlp = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.physics_mlp = nn.Sequential(nn.Linear(2, cond_dim // 2), nn.SiLU(), nn.Linear(cond_dim // 2, cond_dim))

        # 🎯 核心重构：纵向拓扑路径升级为 4 层，通道逐层翻倍拓展深层流形表达
        # 降采样路径 (Encoder)
        self.down1 = ConditionalResBlock(64, 64, cond_dim)
        self.down2 = ConditionalResBlock(64, 128, cond_dim)
        self.down3 = ConditionalResBlock(128, 256, cond_dim)
        self.down4 = ConditionalResBlock(256, 512, cond_dim)  # 新增最底层物理瓶颈层 (Bottleneck)
        
        # 上采样路径 (Decoder)
        self.up1 = ConditionalResBlock(512 + 256, 256, cond_dim)  # 新增对应的第一层解码
        self.up2 = ConditionalResBlock(256 + 128, 128, cond_dim)
        self.up3 = ConditionalResBlock(128 + 64, 64, cond_dim)
        self.up4 = ConditionalResBlock(64 + 64, 64, cond_dim)
        
        self.final_conv = nn.Conv2d(64, flow_ch, 1)

    def forward(self, x_t, grid_x, grid_y, t, physics_cond):
        # 1. 动态对齐空间网格物理维度
        if len(grid_x.shape) == 2:
            grid_x = grid_x.unsqueeze(0).expand(x_t.shape[0], -1, -1)
            grid_y = grid_y.unsqueeze(0).expand(x_t.shape[0], -1, -1)
        if len(grid_x.shape) == 3:
            grid_x = grid_x.unsqueeze(1)
            grid_y = grid_y.unsqueeze(1)

        # 空间感知道路打通：通道硬拼接 [B, 6, H, W]
        x = torch.cat([x_t, grid_x, grid_y], dim=1) 
        orig_h, orig_w = x.shape[2], x.shape[3]

        # 2. ⚡ 几何空间安全垫升级 (4层网络要求尺寸必须能被 2^3=8 整除，防下采样非对称奇数截断)
        # 针对 (60, 301) -> H填充4像素变成64; W填充3像素变成304
        pad_h = (8 - orig_h % 8) % 8
        pad_w = (8 - orig_w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

        # 3. 外部宏观控制工况嵌入
        cond = self.time_mlp(get_timestep_embedding(t, 128)) + self.physics_mlp(physics_cond)
        
        # 4. 正向深度特征编码 (4-Layer Encoder)
        x0 = self.init_conv(x)
        d1 = self.down1(x0, cond)
        
        d2_in = F.max_pool2d(d1, 2)
        d2 = self.down2(d2_in, cond)
        
        d3_in = F.max_pool2d(d2, 2)
        d3 = self.down3(d3_in, cond)
        
        d4_in = F.max_pool2d(d3, 2)  # 再次降采样
        d4 = self.down4(d4_in, cond) # 进入最深层特征空间
        
        # 5. 反向自适应尺寸对齐解码 (4-Layer Decoder + Skip Connection)
        # 显式动态锁定上一层的真实输入高宽，杜绝任何单像素错位
        u1_up = F.interpolate(d4, size=(d3.shape[2], d3.shape[3]), mode='nearest')
        u1 = self.up1(torch.cat([u1_up, d3], dim=1), cond)
        
        u2_up = F.interpolate(u1, size=(d2.shape[2], d2.shape[3]), mode='nearest')
        u2 = self.up2(torch.cat([u2_up, d2], dim=1), cond)
        
        u3_up = F.interpolate(u2, size=(d1.shape[2], d1.shape[3]), mode='nearest')
        u3 = self.up3(torch.cat([u3_up, d1], dim=1), cond)
        
        u4 = self.up4(torch.cat([u3, x0], dim=1), cond)
        out = self.final_conv(u4)
        
        # 6. ⚡ 物理特征逆流剪裁：切除刚才动态垫高的像素，毫秒级无损交还原图尺寸
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :orig_h, :orig_w]
        return out




# #usr/bin/python3
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# def get_timestep_embedding(timesteps, embedding_dim):
#     half_dim = embedding_dim // 2
#     emb = math.log(10000) / (half_dim - 1)
#     emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
#     emb = timesteps[:, None] * emb[None, :]
#     return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

# class AdaIN(nn.Module):
#     def __init__(self, cond_dim, channels):
#         super().__init__()
#         self.instance_norm = nn.InstanceNorm2d(channels, affine=False)
#         self.fc = nn.Linear(cond_dim, channels * 2)

#     def forward(self, x, cond):
#         x_norm = self.instance_norm(x)
#         gamma_beta = self.fc(cond).unsqueeze(-1).unsqueeze(-1)
#         gamma, beta = gamma_beta.chunk(2, dim=1)
#         return (1 + gamma) * x_norm + beta

# class ConditionalResBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, cond_dim):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
#         self.adain1 = AdaIN(cond_dim, out_channels)
#         self.act = nn.SiLU()
#         self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
#         self.adain2 = AdaIN(cond_dim, out_channels)
#         self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

#     def forward(self, x, cond):
#         h = self.adain1(self.conv1(x), cond)
#         h = self.adain2(self.conv2(self.act(h)), cond)
#         return h + self.shortcut(x)

# class DiffusionUNet(nn.Module):
#     """
#     针对 64x64 稳态流场自适应对齐的轻量紧凑型去噪网络
#     """
#     def __init__(self, flow_ch=4, coord_ch=2, cond_dim=128):
#         super().__init__()
#         self.init_conv = nn.Conv2d(flow_ch + coord_ch, 64, 3, padding=1)
        
#         self.time_mlp = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
#         self.physics_mlp = nn.Sequential(nn.Linear(2, cond_dim // 2), nn.SiLU(), nn.Linear(cond_dim // 2, cond_dim))

#         # 拓扑路径：64x64 -> 32x32 -> 16x16 -> 8x8
#         self.down1 = ConditionalResBlock(64, 64, cond_dim)
#         self.down2 = ConditionalResBlock(64, 128, cond_dim)
#         self.down3 = ConditionalResBlock(128, 256, cond_dim)
        
#         self.up1 = ConditionalResBlock(256 + 128, 128, cond_dim)
#         self.up2 = ConditionalResBlock(128 + 64, 64, cond_dim)
#         self.up3 = ConditionalResBlock(64 + 64, 64, cond_dim)
        
#         self.final_conv = nn.Conv2d(64, flow_ch, 1)

#     def forward(self, x_t, grid_x, grid_y, t, physics_cond):
#         # 接收并拼装 64x64 空间坐标
#         x = torch.cat([x_t, grid_x.unsqueeze(1), grid_y.unsqueeze(1)], dim=1) 
        
#         cond = self.time_mlp(get_timestep_embedding(t, 128)) + self.physics_mlp(physics_cond)
        
#         x0 = self.init_conv(x)
#         d1 = self.down1(x0, cond)
#         d2 = self.down2(F.max_pool2d(d1, 2), cond)
#         d3 = self.down3(F.max_pool2d(d2, 2), cond)
        
#         u1 = self.up1(torch.cat([F.interpolate(d3, scale_factor=2, mode='nearest'), d2], dim=1), cond)
#         u2 = self.up2(torch.cat([F.interpolate(u1, scale_factor=2, mode='nearest'), d1], dim=1), cond)
#         u3 = self.up3(torch.cat([u2, x0], dim=1), cond)
        
#         return self.final_conv(u3)



