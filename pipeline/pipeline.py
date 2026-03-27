"""
pipeline/pipeline.py
====================
核心流水线：串联去模糊（DeepDeblur）与目标检测（CEASC）。

流程:
    输入模糊图像
        → DeblurModel.deblur()   → 清晰图像
        → CEASCDetector.detect() → DetectionResult
        → Visualizer.draw()      → 可视化结果图

典型用法:
    pipe = DeblurDetPipeline.from_config("configs/pipeline.yaml")
    result = pipe.run(blurry_image)
    pipe.visualize(result, save_path="output.jpg")
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """流水线完整输出。"""
    blurry_image:   np.ndarray      # 原始模糊图像
    sharp_image:    np.ndarray      # 去模糊后图像
    detection:      "DetectionResult"  # 检测结果（类型引用，避免循环导入）
    deblur_time:    float           # 去模糊耗时（秒）
    detect_time:    float           # 检测耗时（秒）

    @property
    def total_time(self) -> float:
        return self.deblur_time + self.detect_time

    @property
    def num_objects(self) -> int:
        return self.detection.num_objects


class DeblurDetPipeline:
    """
    去模糊 + 目标检测端到端流水线。

    Args:
        deblur_model:  DeblurModel 实例
        detector:      CEASCDetector 实例
        skip_deblur:   若为 True，跳过去模糊直接检测（用于消融对比）
    """

    def __init__(self, deblur_model, detector, skip_deblur: bool = False):
        self.deblur_model = deblur_model
        self.detector = detector
        self.skip_deblur = skip_deblur

    @classmethod
    def from_config(
        cls,
        deblur_checkpoint: str,
        det_config: str,
        det_checkpoint: str,
        device: str = "auto",
        tile_size: int = 0,
        score_thr: float = 0.3,
        nms_thr: float = 0.5,
        skip_deblur: bool = False,
    ) -> "DeblurDetPipeline":
        """从权重路径直接构建流水线（推荐入口）。"""
        from deblur import DeblurModel
        from detection import CEASCDetector

        logger.info("正在加载去模糊模型...")
        deblur = DeblurModel(
            checkpoint=deblur_checkpoint,
            device=device,
            tile_size=tile_size,
        )

        logger.info("正在加载目标检测模型...")
        detector = CEASCDetector(
            config=det_config,
            checkpoint=det_checkpoint,
            device=device,
            score_thr=score_thr,
            nms_thr=nms_thr,
        )

        return cls(deblur, detector, skip_deblur=skip_deblur)

    def run(self, image: np.ndarray) -> PipelineResult:
        """
        对单张图像运行完整流水线。

        Args:
            image: BGR numpy 数组，uint8
        Returns:
            PipelineResult，包含中间结果和最终检测框
        """
        # ── 阶段 1：去模糊 ──────────────────────────────
        t0 = time.perf_counter()
        if self.skip_deblur:
            sharp = image.copy()
            logger.debug("跳过去模糊（skip_deblur=True）")
        else:
            sharp = self.deblur_model.deblur(image)
        deblur_time = time.perf_counter() - t0

        # ── 阶段 2：目标检测 ────────────────────────────
        t1 = time.perf_counter()
        detection = self.detector.detect(sharp)
        detect_time = time.perf_counter() - t1

        logger.info(
            f"流水线完成 | 去模糊: {deblur_time:.2f}s | "
            f"检测: {detect_time:.2f}s | 目标数: {detection.num_objects}"
        )

        return PipelineResult(
            blurry_image=image,
            sharp_image=sharp,
            detection=detection,
            deblur_time=deblur_time,
            detect_time=detect_time,
        )

    def run_batch(self, images: list) -> list:
        """批量处理多张图像。"""
        return [self.run(img) for img in images]

    def run_on_file(self, input_path: Union[str, Path], save_dir: Optional[str] = None) -> PipelineResult:
        """从文件路径读取图像并运行流水线，可选保存结果。"""
        input_path = Path(input_path)
        image = cv2.imread(str(input_path))
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {input_path}")

        result = self.run(image)

        if save_dir:
            from pipeline.visualizer import Visualizer
            vis = Visualizer()
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            # 保存可视化结果
            vis_img = vis.draw(result)
            cv2.imwrite(str(save_dir / f"{input_path.stem}_result.jpg"), vis_img)

            # 保存去模糊图
            cv2.imwrite(str(save_dir / f"{input_path.stem}_sharp.jpg"), result.sharp_image)

            logger.info(f"结果保存至: {save_dir}")

        return result
