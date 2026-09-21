# 2026-09-21 点云迁移与本地整理

## 可恢复归档

本地归档：`D:/Project/PhotoAgent-archive/20260921-pointcloud-migration/`。
`before/` 是修改前的源码、测试、实验、文档、示例和依赖快照，包含此前未提交的实验代码。
`retired/` 保留本次从活动目录移出的原文件，未永久删除；目录相对路径与原 PhotoAgent 一致。
归档不属于 Git 仓库，不参与导入、测试、打包或提交。
需要恢复某项时，从 before/ 或 retired/ 复制对应路径到新目录检查，再选择性合并；不要覆盖新代码整棵树。

## 已移出活动目录

- Homography/viewport 2D renderer、Mesh renderer、旧单 K point warper 与 renderer dispatch。
- YOLO subject detection、mask/inpaint/copy-paste/reposition 模块和 YOLO 权重。
- camera_translation（含 up_pitch）、viewpoint_rotation、fit_target_state、unified_camera_rendering 四组实验。
- 上述旧实现专属的测试、Mesh 安装文件、过期说明和 20 份旧示例。
- 旧 build/ 打包副本；workdir 下 40 项历史 demo、验收、单例实验等产物。
- PhotoAgent 内重复的 target_sketch_viewer/，完整内容归档，独立仓库继续使用 D:/Project/viewer。

原有 .venv / .venv311 环境和 data/ 输入保留，未批量删除未知用途的数据或其他项目。
requirements.txt 及历史 lock 移除 Ultralytics 三个包；已有环境中的安装不主动卸载。
Depth Pro 保留为可选深度估计器，不再与 Mesh 绑定。

## 当前保留

- 唯一实验入口：experiments/depth_pointcloud_sketch/，与正式渲染共用同一 point renderer。
- workdir/depth_models/：现有模型缓存。
- workdir/depth_pointcloud_sketch_20260920/：27 张图、270 个旧基准结果和审计记录。
- workdir/pointcloud_migration_20260921/：本次 batch、正式 workflow、对比和 Guidance 验收。
- workdir/viewer_data_20260921.zip：原可迁移 Viewer 数据包。
- Viewer 仓库 D:/Project/viewer，8080 服务已改为从此目录启动，页面/图片 HTTP 200。

这次更改留在 PhotoAgent working tree，未 commit、未 push；Viewer 远端代码未改动。

## 验收

- `python -m pytest tests experiments/depth_pointcloud_sketch/test_batch.py -q`：98 项通过。
- 真实输入 video12_final_01_gpt.png，Depth Anything V2 Small 本地权重 CPU 推理一次，10 个 preset 全部成功。
- 新 batch 与保留的对应 10 个旧 PNG 逐像素相同（RGB MAE=0），覆盖率约 75.00%–99.84%。
- 正式 TargetSketchRenderer 使用同图同深度再跑 10 个 preset，RGB 和 mask 与 batch 完全相同。
- 真实视频 1267846028.mp4：上移 .05 + 俯拍 8°，正式 workflow 验证通过。
- 同一真实视频的 Integrated workflow 到 Guidance 正常持久化；复用上一轮实际估计的深度，LocalBackend 返回 uncertain。
- 未进行新在线 VLM 调用或服务器远端部署。保留的 VLM 配置/传输/重试/检查点测试通过。
- 真实图孔洞和边缘 artifacts 仍保留，未做补全或视觉优化。

可复现命令（输出目录需尚不存在）：

```powershell
.venv311/Scripts/python.exe experiments/depth_pointcloud_sketch/run_batch.py --input-dir "D:/Data/videoagent/test_data/PhotoAgent/data/test_data/gpt_enhanced" --output-dir workdir/new_batch --max-images 1 --depth-model workdir/depth_models/depth-anything-v2-small --device cpu --save-depth --save-debug
.venv311/Scripts/python.exe tools/verify_pointcloud_migration.py --batch-dir workdir/new_batch --output-dir workdir/new_comparison
.venv311/Scripts/python.exe app.py --video "D:/Data/videoagent/test_data/PhotoAgent/1267846028.mp4" --target-state examples/camera_up_pitch_down.json --work-dir workdir/new_production --depth-model workdir/depth_models/depth-anything-v2-small --target-width 640 --target-height 480 --debug
```
