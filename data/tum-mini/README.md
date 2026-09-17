# TUM 小批量 RGB 视频测试集

来源：[TUM RGB-D 官方下载页](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)，下载日期：2026-09-15。
选取官网提供的 RGB movie AVI，原样保存；未下载深度、ROS bag 或完整 RGB-D 序列。

| 文件 | 场景 | 实测时长 | 文件大小 |
|---|---|---:|---:|
| `rgbd_dataset_freiburg1_xyz-rgb.avi` | 桌面场景，主要为相机平移 | 26.6 秒 | 8.06 MB |
| `rgbd_dataset_freiburg1_rpy-rgb.avi` | 桌面场景，主要为原地旋转 | 24.1 秒 | 7.41 MB |
| `rgbd_dataset_freiburg1_room-rgb.avi` | 办公室环游 | 45.4 秒 | 13.77 MB |

共 3 个视频，29,236,350 字节（约 29.24 MB），均为 640×480、30 FPS。
时长由实际 AVI 帧数计算，可能与官网完整原始序列的采集时长不同。
这些 AVI 可用于当前工程的功能测试；不包含逐帧时间戳、深度或相机位姿，不能直接用于位姿真值评测。

## 使用

在本地 Web 页面选择本目录任意 AVI 即可。或者在工程根目录执行：

```powershell
python -m video_guide demo data/tum-mini/rgbd_dataset_freiburg1_xyz-rgb.avi --seed 42 --end-frame 299
python -m video_guide demo data/tum-mini/rgbd_dataset_freiburg1_xyz-rgb.avi --seed 42 --end-frame 299 --force-current
```

## 验证记录

- 三个文件已逐帧完整解码，解码帧数与文件报告帧数一致。
- 各使用前 300 帧、种子 42 验证随机目标和强制当前目标两种情况，共 6 次本地后端运行。
- 三次随机目标运行进入导航分支；三次强制目标运行进入构图分支并生成裁剪图。
- 此验证检查数据可用性和流程，不代表移动建议经过位姿真值评估。
- `manifest.json` 保存来源 URL、大小、SHA-256、视频参数和验证结果。验证输出位于工程根目录下的 `outputs/<task_id>/run/`。

视频文件已加入 Git 忽略规则，元数据及本文档可纳入版本管理。使用及引用数据请遵循官方数据集说明。

历史测试结果已按四项输入文件哈希关联，迁移到 `outputs/<task_id>/pre` 和 `run`；历史结果保留旧版本输入与输出。
