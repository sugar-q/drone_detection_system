"""
detection/detector.py
=====================
CEASC 目标检测的推理封装。

基于: https://github.com/Cuogeihong/CEASC
论文: Adaptive Sparse Convolutional Networks with Global Context Enhancement
      for Faster Object Detection on Drone Images (CVPR 2023)

使用方法:
    detector = CEASCDetector(
        config="configs/ceasc_gfl_res18_visdrone.py",
        checkpoint="weights/detect/ceasc_gfl_visdrone.pth"
    )
    results = detector.detect(image)   # numpy BGR → DetectionResult
"""

import sys
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# VisDrone 类别映射（10 类）
VISDRONE_CLASSES = (
    "pedestrian", "people", "bicycle", "car",
    "van", "truck", "tricycle", "awning-tricycle",
    "bus", "motor",
)

# 中文类别名（用于可视化）
VISDRONE_CLASSES_ZH = (
    "行人", "人群", "自行车", "汽车",
    "面包车", "卡车", "三轮车", "遮阳三轮车",
    "公交车", "摩托车",
)


@dataclass
class DetectionResult:
    """单张图像的检测结果。"""
    boxes:      np.ndarray          # (N, 4) xyxy 格式
    scores:     np.ndarray          # (N,) 置信度
    labels:     np.ndarray          # (N,) 类别 ID
    class_names: List[str] = field(default_factory=list)   # 类别名

    @property
    def num_objects(self) -> int:
        return len(self.boxes)

    def filter_by_score(self, threshold: float) -> "DetectionResult":
        """按置信度阈值过滤。"""
        mask = self.scores >= threshold
        return DetectionResult(
            boxes=self.boxes[mask],
            scores=self.scores[mask],
            labels=self.labels[mask],
            class_names=self.class_names,
        )

    def to_dict(self) -> dict:
        return {
            "boxes":  self.boxes.tolist(),
            "scores": self.scores.tolist(),
            "labels": self.labels.tolist(),
            "class_names": [self.class_names[l] for l in self.labels],
        }


class CEASCDetector:
    """
    CEASC 无人机目标检测推理接口。

    依赖 MMDetection 和 CEASC 子项目。
    安装步骤见 README.md。

    Args:
        config:      CEASC 配置文件路径（.py）
        checkpoint:  预训练权重路径（.pth）
        device:      'cuda:0' | 'cpu' | 'auto'
        score_thr:   默认置信度阈值
        nms_thr:     NMS 阈值
    """

    def __init__(
        self,
        config: str,
        checkpoint: str,
        device: str = "auto",
        score_thr: float = 0.3,
        nms_thr: float = 0.5,
    ):
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.class_names = list(VISDRONE_CLASSES)

        # 自动选择设备
        if device == "auto":
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

        # 将 CEASC third_party 加入路径
        self._setup_ceasc_path()

        # 加载 MMDetection 模型
        self.model = self._build_model(config, checkpoint)
        self._override_test_cfg()
        logger.info(f"CEASCDetector 已加载: {checkpoint}")

    def _setup_ceasc_path(self):
        """确保 CEASC 和 MMDet 可被导入。"""
        ceasc_path = Path(__file__).parents[1] / "third_party" / "CEASC"
        if ceasc_path.exists() and str(ceasc_path) not in sys.path:
            sys.path.insert(0, str(ceasc_path))

    def _build_model(self, config: str, checkpoint: str):
        """使用 MMDetection API 构建模型。"""
        try:
            from mmdet.apis import init_detector  # type: ignore
        except ImportError:
            raise ImportError(
                "请先安装 mmdet: pip install mmdet==2.24.1\n"
                "并完成 CEASC 稀疏卷积算子安装:\n"
                "  cd third_party/CEASC/Sparse_conv && python setup.py install"
            )

        model = init_detector(config, checkpoint, device=self.device)
        return model

    def _override_test_cfg(self) -> None:
        """Apply score/NMS overrides to the underlying config if possible."""
        cfg = getattr(self.model, "cfg", None)
        if cfg is None:
            return

        def _apply_to_test_cfg(test_cfg):
            if not isinstance(test_cfg, dict):
                return
            test_cfg["score_thr"] = float(self.score_thr)
            if "nms" in test_cfg and isinstance(test_cfg["nms"], dict):
                test_cfg["nms"]["iou_threshold"] = float(self.nms_thr)

        if "test_cfg" in cfg:
            _apply_to_test_cfg(cfg["test_cfg"])
        elif "model" in cfg and isinstance(cfg["model"], dict) and "test_cfg" in cfg["model"]:
            _apply_to_test_cfg(cfg["model"]["test_cfg"])

    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        对单张图像执行目标检测。

        Args:
            image: BGR numpy 数组，shape (H, W, 3)，dtype uint8
        Returns:
            DetectionResult 对象
        """
        from mmdet.apis import inference_detector  # type: ignore

        raw = inference_detector(self.model, image)
        return self._parse_results(raw)

    def detect_batch(self, images: List[np.ndarray]) -> List[DetectionResult]:
        """批量检测。"""
        return [self.detect(img) for img in images]

    def _parse_results(self, raw_results) -> DetectionResult:
        """将 MMDetection 原始输出解析为 DetectionResult。"""
        all_boxes, all_scores, all_labels = [], [], []

        if isinstance(raw_results, tuple):
            # (bbox_results, segm_results)
            raw_results = raw_results[0]

        for cls_id, bboxes in enumerate(raw_results):
            if len(bboxes) == 0:
                continue
            bboxes = np.array(bboxes)
            scores = bboxes[:, 4]
            boxes = bboxes[:, :4]
            labels = np.full(len(bboxes), cls_id, dtype=np.int64)

            mask = scores >= self.score_thr
            all_boxes.append(boxes[mask])
            all_scores.append(scores[mask])
            all_labels.append(labels[mask])

        if all_boxes:
            boxes  = np.concatenate(all_boxes, axis=0)
            scores = np.concatenate(all_scores, axis=0)
            labels = np.concatenate(all_labels, axis=0)
        else:
            boxes  = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,),   dtype=np.float32)
            labels = np.zeros((0,),   dtype=np.int64)

        return DetectionResult(
            boxes=boxes,
            scores=scores,
            labels=labels,
            class_names=self.class_names,
        )
