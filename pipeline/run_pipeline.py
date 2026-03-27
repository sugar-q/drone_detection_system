#!/usr/bin/env python3
"""
pipeline/run_pipeline.py
========================
命令行入口：端到端模糊无人机图像目标检测。

用法示例:
    # 单张图像
    python pipeline/run_pipeline.py \
        --input blurry.jpg \
        --deblur-checkpoint weights/deblur/DeepDeblur_GOPRO.pt \
        --det-config  configs/ceasc_gfl_res18_visdrone.py \
        --det-checkpoint weights/detect/ceasc_gfl_visdrone.pth \
        --output results/ --vis

    # 批量目录
    python pipeline/run_pipeline.py \
        --input drone_images/ \
        --output results/ --save-deblurred

    # 消融对比（不去模糊，直接检测）
    python pipeline/run_pipeline.py --input img.jpg ... --skip-deblur
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2

# 将项目根目录加入 Python 路径
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.pipeline import DeblurDetPipeline
from pipeline.visualizer import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pipeline")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="DeblurDet: 模糊无人机图像目标检测系统",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 输入 / 输出
    parser.add_argument("--input",  required=True, help="输入图像文件或目录")
    parser.add_argument("--output", default="results/", help="输出目录")

    # 模型权重
    parser.add_argument(
        "--deblur-checkpoint",
        default="weights/deblur/DeepDeblur_GOPRO.pt",
        help="DeepDeblur 预训练权重路径",
    )
    parser.add_argument(
        "--det-config",
        default="configs/ceasc_gfl_res18_visdrone.py",
        help="CEASC 配置文件路径",
    )
    parser.add_argument(
        "--det-checkpoint",
        default="weights/detect/ceasc_gfl_visdrone.pth",
        help="CEASC 预训练权重路径",
    )

    # 设备
    parser.add_argument("--device", default="auto", help="'cuda:0' | 'cpu' | 'auto'")

    # 去模糊选项
    parser.add_argument("--tile-size", type=int, default=0,
                        help="分块推理块大小（0=整图推理，高分辨率建议 512）")
    parser.add_argument("--skip-deblur", action="store_true",
                        help="跳过去模糊，直接检测（消融实验用）")

    # 检测选项
    parser.add_argument("--score-thr", type=float, default=0.3, help="检测置信度阈值")
    parser.add_argument("--nms-thr", type=float, default=0.5, help="NMS IoU 阈值")

    # 输出选项
    parser.add_argument("--vis", action="store_true", help="保存三联可视化对比图")
    parser.add_argument("--save-deblurred", action="store_true", help="保存去模糊中间结果")
    parser.add_argument("--save-json", action="store_true", help="保存 JSON 格式检测结果")

    return parser.parse_args()


def collect_images(input_path: Path) -> list:
    """收集输入路径下的所有图像文件。"""
    if input_path.is_file():
        return [input_path]
    return sorted([
        p for p in input_path.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])


def main():
    args = parse_args()

    input_path  = Path(args.input)
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 构建流水线 ──────────────────────────────────────────
    logger.info("正在初始化流水线...")
    pipeline = DeblurDetPipeline.from_config(
        deblur_checkpoint=args.deblur_checkpoint,
        det_config=args.det_config,
        det_checkpoint=args.det_checkpoint,
        device=args.device,
        tile_size=args.tile_size,
        score_thr=args.score_thr,
        nms_thr=args.nms_thr,
        skip_deblur=args.skip_deblur,
    )
    visualizer = Visualizer()

    # ── 收集图像文件 ────────────────────────────────────────
    image_files = collect_images(input_path)
    if not image_files:
        logger.error(f"未找到图像文件: {input_path}")
        sys.exit(1)

    logger.info(f"共找到 {len(image_files)} 张图像，开始处理...")

    # ── 逐张处理 ───────────────────────────────────────────
    total_objects = 0
    json_all = []

    for idx, img_path in enumerate(image_files, 1):
        logger.info(f"[{idx}/{len(image_files)}] 处理: {img_path.name}")
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning(f"  跳过（无法读取）: {img_path}")
            continue

        result = pipeline.run(image)
        total_objects += result.num_objects

        stem = img_path.stem

        # 保存三联可视化
        if args.vis:
            vis_img = visualizer.draw(result)
            vis_path = output_dir / f"{stem}_comparison.jpg"
            cv2.imwrite(str(vis_path), vis_img)

        # 保存去模糊图
        if args.save_deblurred:
            sharp_path = output_dir / f"{stem}_sharp.jpg"
            cv2.imwrite(str(sharp_path), result.sharp_image)

        # 收集 JSON
        if args.save_json:
            json_all.append({
                "image": str(img_path),
                "deblur_ms":  round(result.deblur_time * 1000, 1),
                "detect_ms":  round(result.detect_time * 1000, 1),
                "num_objects": result.num_objects,
                "detections": result.detection.to_dict(),
            })

        logger.info(
            f"  ✅ 目标数: {result.num_objects} | "
            f"去模糊: {result.deblur_time*1000:.0f}ms | "
            f"检测: {result.detect_time*1000:.0f}ms"
        )

    # 保存全部 JSON
    if args.save_json:
        json_path = output_dir / "detections.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_all, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON 结果已保存: {json_path}")

    logger.info(
        f"\n{'='*50}\n"
        f"  处理完成！共处理 {len(image_files)} 张图像\n"
        f"  总检测目标数: {total_objects}\n"
        f"  结果保存至: {output_dir}\n"
        f"{'='*50}"
    )


if __name__ == "__main__":
    main()
