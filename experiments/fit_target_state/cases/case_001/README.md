# case_001 真实静态图对验收

输入：用户提供的 590×786 reference.jpg / ground_truth.jpg，原始素材未纳入仓库。
本机素材目录：`D:/Data/videoagent/test_data/PhotoAgent/agent_test_data/case_001`。

## 复现

在仓库根目录运行，输出目录必须不存在或为空：

```powershell
.\.venv311\Scripts\python experiments/fit_target_state/run.py `
  --reference D:/Data/videoagent/test_data/PhotoAgent/agent_test_data/case_001/reference.jpg `
  --ground-truth D:/Data/videoagent/test_data/PhotoAgent/agent_test_data/case_001/ground_truth.jpg `
  --config experiments/fit_target_state/cases/case_001/config.json `
  --ignore-mask experiments/fit_target_state/cases/case_001/gt_ignore.png `
  --reference-ignore-mask experiments/fit_target_state/cases/case_001/reference_ignore.png `
  --output-dir workdir/case_001_rerun
```

## 准备与限制

- 原始 case_config 的嵌套字段和像素 ignore_rects 转换到实验配置；原文件保持原样。
- 默认 640 推理在 reference 漏检；1280、confidence=0.2 检出约 20×53 像素人物，置信度 0.310。
- GT 640 默认检测只覆盖头部到字幕上沿，1280 也未解决遮挡。因此人工标注完整 GT 框 `[428,466,524,740]`，在 config 中归一化并显式启用 `use_manual_gt_bbox`。这是带人工辅助的拟合，人物像素误差相对于此框，不是独立检测精度。
- 对两张图分别屏蔽触控圆点、字幕、倍率 UI、网格线；GT 另外屏蔽红色准星。reference mask 跟随 viewport 变换。屏蔽只影响 loss，不清除草图里的原始网格或恢复遮挡内容。
- 收窄准星 mask，保留更多雪山/坡面；有效背景比例下限从 0.15 提高到 0.70，防止扩大 UI 屏蔽区域投机。
- 加入可配置 RGB loss（本例 0.8）区分灰度相近的蓝天和树林。
- 无背景先验的搜索仍偏好小雪山。本例推荐配置以图片的 1×→5× UI 为粗略先验，将 **二维图像缩放** 限制到 3.5～6.5，初始 zoom=5；不建模真实焦距或透视。viewport 中心仍自动搜索，没有人工山峰坐标参与目标函数。
- 原图小人物放大约五倍，模糊、分割边缘、原位置修补痕迹及姿态差异不可避免。GT 被字幕遮挡部分不能验证细节。

## 迭代记录

所有实际运行保存在 `workdir/case_001_acceptance/`：

| 目录 | 设置/结果 |
| --- | --- |
| baseline | 原默认检测，reference 无人物，明确失败并停止 |
| detection_fixed | 修正小人物检测、人工 GT 完整框，背景错选树冠，验收失败 |
| refined | 扩大 zoom 到 8，双侧 UI mask；放大圆点使有效背景仅约15%，虚假低损失，拒绝 |
| final | 有效背景提高到70%；灰度评分仍偏爱天空，拒绝。此目录名称不代表最终推荐 |
| color | RGB评分恢复树林/山坡，但 zoom≈1.69，雪山偏小，未通过背景验收 |
| recommended | 加入明确的缩放先验，最终推荐输出；见下方实际结果 |

不同 loss / mask 的损失值不可直接横向比较。已通过 134 项测试，包含 reference mask 变换、人工 GT override 和近等亮度颜色混淆回归验证。正式 workflow、LangGraph 和视频模块未修改。

## 最终实际结果与验收判断

推荐产物目录：`D:/Project/PhotoAgent/workdir/case_001_acceptance/recommended`，包含 best_target_state.json、best_sketch.jpg、comparison_triptych.jpg、search_log.json。

- 输出 590×786，CPU 耗时约 70.7 秒，3098 个候选（以 search_log 的 evaluations 为准）。二维背景缩放约 3.523×，接近搜索下限，说明当前 loss 仍偏好较小的山峰。
- 人物实际 bbox 为 `[424,463,528,739]`，人工 GT 框为 `[428,466,524,740]`。底部中心横坐标一致，底部高 1 像素，高度 276 vs 274（多 2 像素），宽度 104 vs 96（多 8 像素）。宽高比例来自原人物，姿态变化不能由单纯缩放解决。
- 雪山进入画面左侧并明显放大，右侧树林及人物相对关系大致成立。但雪山仍偏小、偏右下，山坡与云层位置不一致；不能判定背景精确对齐通过。
- 原始人物仅约20×53像素，放大后明显模糊，人物原位置的 inpaint 条带可见；reference 网格线随图像放大，仍影响视觉展示。mask 只能避免评分受其影响，不能恢复干净照片。
- 预览 loss 0.047727，导出复算 0.047947，有效背景约占 GT 可评分区域的 80.3%。这个分母已经排除了 GT UI，并非全图有效比例。
- **结论：带人工 GT 框和缩放先验的粗粒度构图表示部分通过；完全自动背景拟合尚未通过。** 不能用本例证明无先验的泛化能力。
- **下一步建议：暂缓把视频帧对齐作为正式验收。** 先用3～5组无字幕或可可靠屏蔽的真实静态图对，验证山峰/树木等背景位置和尺度；背景评分稳定后再做小规模离线视频帧 vs target sketch 对齐。当前可以作为后续视频探索的基线，但不应拿本例低 loss 当可靠匹配分数。
