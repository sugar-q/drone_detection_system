"""
deblur/deblur_model.py
======================
DeepDeblur-PyTorch 的推理封装。

基于: https://github.com/SeungjunNah/DeepDeblur-PyTorch
论文: Deep Multi-scale CNN for Dynamic Scene Deblurring (CVPR 2017)

使用方法:
    model = DeblurModel(checkpoint="weights/deblur/DeepDeblur_GOPRO.pt")
    sharp = model.deblur(blurry_image)   # numpy HxWx3 uint8 → numpy HxWx3 uint8
"""

import sys
import os
import logging
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# DeepDeblur 网络结构（精简移植自 DeepDeblur-PyTorch）
# 完整实现见: third_party/DeepDeblur-PyTorch/src/model/dmphn.py
# ──────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """残差块"""
    def __init__(self, channels: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.body(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, num_features: int = 64, num_resblocks: int = 8):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResBlock(num_features) for _ in range(num_resblocks)])

    def forward(self, x):
        feat = self.head(x)
        return feat + self.body(feat)


class Decoder(nn.Module):
    def __init__(self, num_features: int = 64, out_channels: int = 3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, 1),
            nn.ReLU(inplace=True),
            ResBlock(num_features),
            ResBlock(num_features),
            nn.Conv2d(num_features, out_channels, 3, 1, 1),
        )

    def forward(self, feat_low, feat_high):
        feat_low_up = F.interpolate(feat_low, size=feat_high.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([feat_low_up, feat_high], dim=1)
        return self.body(x)


class DeepDeblurNet(nn.Module):
    """
    Multi-scale DeepDeblur 网络（3 尺度）。
    若已安装 third_party/DeepDeblur-PyTorch，会优先加载原始实现。
    """
    NUM_SCALES = 3

    def __init__(self, num_features: int = 64, num_resblocks: int = 8):
        super().__init__()
        self.encoders = nn.ModuleList([
            Encoder(3, num_features, num_resblocks) for _ in range(self.NUM_SCALES)
        ])
        self.decoders = nn.ModuleList([
            Decoder(num_features, 3) for _ in range(self.NUM_SCALES - 1)
        ])
        self.final_conv = nn.Conv2d(num_features, 3, 3, 1, 1)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) 归一化到 [0, 1]
        Returns:
            sharp: (B, 3, H, W) 去模糊结果
        """
        h, w = x.shape[-2:]
        # 构建图像金字塔
        pyramid = [x]
        for _ in range(self.NUM_SCALES - 1):
            pyramid.append(F.interpolate(pyramid[-1], scale_factor=0.5, mode='bilinear', align_corners=False))
        pyramid = pyramid[::-1]  # 从小到大

        # 逐尺度编码
        features = [enc(img) for enc, img in zip(self.encoders, pyramid)]

        # 从最小尺度逐步解码融合
        out = features[0]
        for i, dec in enumerate(self.decoders):
            delta = dec(out, features[i + 1])
            out = features[i + 1] + delta

        sharp = self.final_conv(out)
        return torch.clamp(x + sharp, 0.0, 1.0)


# ──────────────────────────────────────────────────────────────
# 公共接口
# ──────────────────────────────────────────────────────────────

class DeblurModel:
    """
    去模糊推理接口。

    Args:
        checkpoint: 预训练权重路径 (.pt / .pth)
        device:     'cuda' | 'cpu' | 'auto'
        tile_size:  分块推理的块大小（0 = 不分块，高分辨率图建议 512）
        tile_overlap: 分块重叠像素
    """

    def __init__(
        self,
        checkpoint: str,
        device: str = "auto",
        tile_size: int = 0,
        tile_overlap: int = 32,
    ):
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

        # 自动选择设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 尝试从 third_party 加载完整模型
        self.model = self._load_model(checkpoint)
        self.model.eval()
        logger.info(f"DeblurModel 已加载: {checkpoint} → 设备: {self.device}")

    def _load_model(self, checkpoint: str) -> nn.Module:
        """加载模型权重。优先使用 third_party 中的原始实现。"""
        third_party_path = Path(__file__).parents[1] / "third_party" / "DeepDeblur-PyTorch" / "src"
        if third_party_path.exists():
            sys.path.insert(0, str(third_party_path))
            try:
                from model.MPRNet import MPRNet  # type: ignore
                model = MPRNet()
                logger.info("使用 third_party/DeepDeblur-PyTorch 的完整实现")
            except ImportError:
                model = DeepDeblurNet()
                logger.warning("无法导入 third_party 模型，使用内置简化版")
        else:
            model = DeepDeblurNet()
            logger.warning("未找到 third_party/DeepDeblur-PyTorch，使用内置简化版")

        # 加载权重
        ckpt = torch.load(checkpoint, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
        # 去除 'module.' 前缀（DataParallel 残留）
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])
        if missing or unexpected:
            logger.warning(
                "Checkpoint keys mismatch: missing=%d, unexpected=%d",
                len(missing), len(unexpected)
            )
        return model.to(self.device)

    @torch.no_grad()
    def deblur(self, image: np.ndarray) -> np.ndarray:
        """
        对单张图像去模糊。

        Args:
            image: BGR numpy 数组，shape (H, W, 3)，dtype uint8
        Returns:
            sharp: BGR numpy 数组，shape (H, W, 3)，dtype uint8
        """
        # BGR → RGB，归一化
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)

        if self.tile_size > 0:
            output = self._tile_infer(tensor)
        else:
            output = self.model(tensor)

        # 转回 numpy BGR
        out_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_np = np.clip(out_np * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

    @torch.no_grad()
    def deblur_batch(self, images: list) -> list:
        """批量去模糊。"""
        return [self.deblur(img) for img in images]

    def _tile_infer(self, tensor: torch.Tensor) -> torch.Tensor:
        """分块推理，避免高分辨率图像显存不足。"""
        B, C, H, W = tensor.shape
        s = self.tile_size
        ov = self.tile_overlap
        output = torch.zeros_like(tensor)
        weight = torch.zeros(B, 1, H, W, device=self.device)

        for y in range(0, H, s - ov):
            for x in range(0, W, s - ov):
                y1, y2 = y, min(y + s, H)
                x1, x2 = x, min(x + s, W)
                tile = tensor[:, :, y1:y2, x1:x2]
                result = self.model(tile)
                output[:, :, y1:y2, x1:x2] += result
                weight[:, :, y1:y2, x1:x2] += 1.0

        return output / weight.clamp(min=1.0)
