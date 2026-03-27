# 修改说明

## 我做了什么
- 将检测阈值从 CLI 传入流水线，确保 `--score-thr` 和新增的 `--nms-thr` 真正影响推理结果。
- 在可用时覆盖 MMDetection 的测试配置，应用 `score_thr` 与 `nms_thr`。
- 增加去模糊权重与模型结构不匹配时的告警，避免静默加载失败。

## 修改的文件
- pipeline/pipeline.py
- pipeline/run_pipeline.py
- detection/detector.py
- deblur/deblur_model.py

## 备注
- 如果看到去模糊模型有 missing/unexpected keys 警告，通常表示权重与模型结构不匹配。
- 可用 `--skip-deblur` 做消融对比，确认去模糊是否提升检测效果。

## 验证步骤（完整流程）

### 1. 单张图像基线（跳过去模糊）
```bash
python pipeline/run_pipeline.py \
	--input path/to/blurry_image.jpg \
	--output results_baseline/ \
	--deblur-checkpoint weights/deblur/DeepDeblur_GOPRO.pt \
	--det-config configs/ceasc_gfl_res18_visdrone.py \
	--det-checkpoint weights/detect/ceasc_visdrone.pth \
	--skip-deblur \
	--score-thr 0.3 \
	--nms-thr 0.5 \
	--vis
```

### 2. 单张图像完整链路（去模糊 + 检测）
```bash
python pipeline/run_pipeline.py \
	--input path/to/blurry_image.jpg \
	--output results_deblur/ \
	--deblur-checkpoint weights/deblur/DeepDeblur_GOPRO.pt \
	--det-config configs/ceasc_gfl_res18_visdrone.py \
	--det-checkpoint weights/detect/ceasc_visdrone.pth \
	--score-thr 0.3 \
	--nms-thr 0.5 \
	--vis \
	--save-deblurred
```

对比两次输出的检测目标数与可视化结果，判断去模糊是否带来改善。

### 3. 阈值敏感性测试（可选）
```bash
python pipeline/run_pipeline.py \
	--input path/to/blurry_image.jpg \
	--output results_thr_010/ \
	--deblur-checkpoint weights/deblur/DeepDeblur_GOPRO.pt \
	--det-config configs/ceasc_gfl_res18_visdrone.py \
	--det-checkpoint weights/detect/ceasc_visdrone.pth \
	--score-thr 0.1 \
	--nms-thr 0.5 \
	--vis

python pipeline/run_pipeline.py \
	--input path/to/blurry_image.jpg \
	--output results_thr_030/ \
	--deblur-checkpoint weights/deblur/DeepDeblur_GOPRO.pt \
	--det-config configs/ceasc_gfl_res18_visdrone.py \
	--det-checkpoint weights/detect/ceasc_visdrone.pth \
	--score-thr 0.3 \
	--nms-thr 0.5 \
	--vis
```

观察召回率与误检的变化，确定合适的 `score_thr` 与 `nms_thr`。
