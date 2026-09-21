# 单图 Depth + Point Cloud 批量实验

这是正式 point-cloud renderer 的批量入口；不维护另一套渲染分支。
每张 source 图运行 presets.json 中 10 个 preset，无 target image、无人物编辑或生成补全。

```sh
python experiments/depth_pointcloud_sketch/run_batch.py --input-dir /path/to/images --output-dir workdir/batch-001 --device cpu --save-depth --save-debug
python experiments/depth_pointcloud_sketch/verify_outputs.py workdir/batch-001
python tools/verify_pointcloud_migration.py --batch-dir workdir/batch-001 --output-dir workdir/production-vs-batch
```

--max-images 1 可做 smoke；--depth-model 支持本地模型目录；--presets 指定配置；--blank-color R G B 指定留白色。
输入支持 jpg/jpeg/png/webp，读取 EXIF 方向。输出目录必须是新目录，单个失败不会中断其他 case。
默认 Depth Anything V2 Small，一次加载权重，每张图一次深度推理、一次 prepare_point_cloud，10 个 preset 和 debug identity 共用不可变点云。
--device cuda 只用于深度推理；点云 splatting 当前在 CPU。

## 相机参数

+x 相机右移、+y 上移、+z 前进，单位为场景中位深度比例。
+yaw 右转、+pitch 仰拍、+roll 顺时针。源水平 FOV 为 60°，变化 K_target 不改变 K_source。
全部正式 preset 的平移非零；identity 仅作为 debug 检查，不计入 10 个效果。
所有参数见 presets.json，没有自动选择 Homography 或渲染后 resize。

## 输出

每图 source.png；depth/depth.npy、depth.json、depth_vis.png；
crop / expand / viewpoint 下每个 preset 都有 result.png、compare.png、valid_mask.png、metadata.json。
summary/all_results_grid.jpg 和 all_comparisons_grid.jpg 只汇总当前 source。
根目录 experiment_summary.json 保存配置、源列表、成功/失败记录和运行耗时。无跨场景 overview。
result.png 原生尺寸且无文字；compare.png 为像素完整的 Source | Result，标签在图像之外。
只有总览缩略图会缩放。--save-debug 保存 identity 和 projected_depth 数组。
coverage 只表示观察到的投影面积，不是视觉质量指标。

## 回归与历史结果

```sh
python -m pytest tests/test_pointcloud_pipeline.py experiments/depth_pointcloud_sketch/test_batch.py -q
```

2026-09-20 的 27 图 / 270 结果保留在 workdir/depth_pointcloud_sketch_20260920，
用于历史比较；其元数据记录当时的旧 point warper / explicit-K adapter，不能误认为是本次新代码运行。
本次迁移重新执行一张真实输入的 10 个 preset，深度实际推理一次，10/10 成功：
workdir/pointcloud_migration_20260921/batch。
该图的新结果与旧 10 个结果逐像素一致；同一深度再通过正式 TargetSketchRenderer 运行，
10 个 RGB 和 mask 也与新 batch 完全一致，报告见 production_vs_batch/verification.json。
