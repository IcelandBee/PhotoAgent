# depth_mesh：Depth Pro + 深度边缘感知2.5D网格

## 状态：开发完成，GPU效果验收待执行

本机 Python 3.11.16 / PyTorch 2.14.0+cpu / CUDA不可用 / Transformers 4.57.6；未安装PyTorch3D。没有升级或替换现有torch。Depth Pro API可导入；已用真实HF后处理器和合成模型输出验证metric存储，**未运行Depth Pro真实权重推理，也未运行PyTorch3D GPU光栅化**。

CPU测试专用的 CpuTriangleOracle 仅用于小型synthetic几何/流程测试，不是正式fallback。正式depth_mesh依赖不满足时失败，绝不静默改用点云。

## 架构与路由

| viewpoint.mode | backend | 深度 |
| --- | --- | --- |
| none | 原行为 | 不执行 |
| rotation | 原Homography | 不执行 |
| depth_3d | 原PointCloudViewpointWarper / fixed splat / Z-buffer | 默认Depth Anything |
| depth_mesh | MeshViewpointWarper / edge-aware faces / PyTorch3D | 默认Depth Pro |

现有graph保持 planner → 可选detect subject → 可选estimate depth → render → validate。两个depth模式才进入深度节点。`depth_backend=auto` 的默认映射位于配置工厂，不在Graph中写具体模型类；`--depth-backend depth_anything/depth_pro/precomputed` 和 `build_workflow(depth_estimator=...)` 均可覆盖。`--depth-path` 保持优先级，复用旧用法。

Mesh的CUDA/PyTorch3D预检在深度节点调用估计器之前执行，避免无GPU时先下载大模型。预检运行一个微型GPU三角形，检查扩展是否实际支持CUDA。

## Depth Pro / metric storage

采用 [HF DepthProImageProcessor / DepthProForDepthEstimation](https://huggingface.co/docs/transformers/model_doc/depth_pro)，模型 `apple/DepthPro-hf`，启用 `use_fov_model=True`。

原始模型输出不能直接当作Z。使用处理器的 `post_process_depth_estimation(outputs,target_sizes=[(H,W)])`，取得校准且恢复原图尺寸的predicted_depth、focal_length、field_of_view。保存处理后的正米制Z，不做min-max或median归一化。

DepthObservation显式保存：

| 类型 | metric | unit | normalization |
| --- | --- | --- | --- |
| 旧relative Z | false | relative | median_one |
| Depth Pro | true | meter | none |

`load_depth()` 对metric只验证尺寸/正有限值，不改变尺度。旧relative仍保持原有行为。无效深度保存为0，mesh不为其建面。

每个深度.npy旁保存同名.json完整DepthObservation，含来源、尺度、focal、FoV、推理时间。`--depth-path` 自动读取该sidecar；也可用 `--depth-metadata-path` 指定。迁移到服务器时，应同时复制.npy和.json；内部旧depth_path不会覆盖实际传入的.npy路径。没有sidecar的旧.npy仍解释成relative Z，不能把无sidecar的米制深度误当成relative。

## 内参与位移

优先使用observation.metadata.focal_length_px，记录 `intrinsics_source=depth_pro`；没有estimated focal时按TargetState.horizontal_fov_deg回退，并记录 `manual_fov`。无效focal、已知宽度不匹配明确拒绝，不静默回退。K仍为fx=fy=f、cx=W/2、cy=H/2。

保留源相机x向右、y向下、z向前；正yaw向右、正pitch抬头、正roll相机顺时针。正translation_x/y/z依然表示相机右移/上移/前移。

```text
scene_reference_depth = median(valid source Z)  # 人物修补之前
movement = normalized_translation * scene_reference_depth
C_source = [movement_x, -movement_y, movement_z]
R = Rz(-roll) @ Rx(-pitch) @ Ry(-yaw)
P_target = R @ (P_source - C_source)
```

metadata分别记录translation_normalized、translation_metric_equivalent（metric时为米）、translation_in_depth_units和scene_reference_depth。即10米典型深度下tx=.05为右移0.5米，不是0.05米。

旧点云baseline接到metric输入时仅在内存中将渲染深度除以scene_reference_depth，并采用同一estimated focal。磁盘metric深度不变。这让同一Depth Pro深度的A/B具有相同K和相对位移；旧Depth Anything输入的像素输出保持兼容。

## 网格与裁面

- 默认mesh_stride=2，可设1；numpy向量化建点/建面，无逐pixel Python循环。
- 每个采样位置统一读取RGB、Z、mask。包含最后一行/列，即使尺寸不能被stride整除。
- 每cell建立两面 `(v00,v01,v10)`、`(v01,v11,v10)`；顶点携带RGB。
- triangle深度差：`max(log(Z)) - min(log(Z))`。大于 `mesh_depth_edge_threshold` 时删除；默认0.12，对应最大/最小深度比约1.127。
- 加入full-resolution相邻深度edge guard，再按stride扩展到采样点，防止薄边界落在采样间隔内而被桥接。这比单看三个顶点更保守。
- 无效Z附近不建面；相机变换后跨越near plane的面保守删除。

reposition保留原人物提取、背景RGB/Z修补、最后subject.bbox合成流程；mesh另对人物mask膨胀3像素，并删除触及遮罩的三角面（连内部也删除）。这比仅切边界更保守，不把原人物深度连接到远背景，也不把修补纹理拉过未知区域。原位置可能留下更大的空洞，属明确取舍。

`mesh_face_mask.png` 是三角形标志图：每cell横向两个像素，分别对应两个面；白=保留、黑=删除。`mesh_valid_vertex_mask.png` 是采样网格的有效Z标志。尺寸不同于原图，不能当作同尺寸图像mask直接使用。

## GPU rasterization

使用 [PyTorch3D rasterize_meshes](https://pytorch3d.readthedocs.io/en/latest/modules/renderer/mesh/rasterize_meshes.html)、Meshes、interpolate_face_attributes；不自写CUDA。

- NDC +X向左、+Y向上；短边范围[-1,1]，长边按宽高比扩展。
- 从源像素整数中心转换：`x_ndc=(W-1-2u)/min(W,H)`，y同理。保留目标camera Z，供透视校正与遮挡排序。
- faces_per_pixel=1，perspective_correct=True，双面光栅化，RGB做透视正确的重心插值。
- PyTorch3D零blur严格排除三角形边，可能让像素中心恰落在共享边时缺失；本实现用1e-12 NDC²的微小容差和clamped barycentrics，仅处理数值接缝。在1080短边下仅约0.00054像素，不用于填disocclusion。
- valid_mask严格来自pix_to_face>=0。无几何区固定fill_color；depth_mesh明确拒绝replicate。

没有safe crop、auto zoom、补裁切、生成式补图、未知区大面积图像修复。后续只执行Agent明确指定的reference_viewport（沿用已有长宽比适配）。

## NVIDIA服务器安装

以下为在服务器**已有项目Python环境**中编译的步骤，不另建conda环境，不预设或替换torch。当前本机CPU torch版本不意味着服务器PyTorch3D兼容性已验证。

先检查驱动、toolkit、当前torch；nvcc toolkit需与torch CUDA构建匹配，并具有C++编译器。PyTorch3D二进制/源码须与当前torch/CUDA配套；官方 [INSTALL.md](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) 的版本列表可能滞后。本次固定源码commit仅用于可复现候选构建，不代表已验证该服务器组合。

```bash
nvidia-smi
nvcc --version
python -c "import sys,torch; print(sys.version); print(torch.__version__,torch.version.cuda,torch.cuda.is_available())"
# 固定已装torch/torchvision，阻止pip隐式替换；冲突时停止并选择兼容组合。
python -c "import importlib.metadata as m; from pathlib import Path; names=['torch','torchvision']; Path('/tmp/photoagent-torch-constraints.txt').write_text('\n'.join(n+'=='+m.version(n) for n in names)+'\n')"
python -m pip install -c /tmp/photoagent-torch-constraints.txt -r requirements.txt -r requirements-mesh.txt
# 要求本机已安装匹配的CUDA toolkit；FORCE_CUDA不是驱动/toolkit安装器。
FORCE_CUDA=1 MAX_JOBS=4 python -m pip install --no-build-isolation --no-deps 'git+https://github.com/facebookresearch/pytorch3d.git@978cd99221b9e0a6a568f1d427854d73363265cf'
python -c "from photography_viewpoint_agent.renderer.mesh_warp import PyTorch3DRasterizer; PyTorch3DRasterizer().check_available(); print('CUDA rasterizer ready')"
python -m pytest tests/test_mesh_gpu.py -q -rs
```

requirements-mesh.txt固定已检查可用的Transformers 4.57.6，加构建辅助依赖；不包含torch或PyTorch3D盲装项。无需升级本机已有Transformers。若编译不兼容，先解决服务器torch/CUDA/PyTorch3D组合，不要在CPU项目里强行升级torch。

## 完整workflow运行

将video路径替换为服务器实际文件。输出目录必须为空。

```bash
python app.py --video /data/ski.mp4 \
  --target-state examples/61_depthmesh__translate_right_small.json \
  --depth-backend depth_pro --depth-model apple/DepthPro-hf \
  --depth-device cuda --mesh-device cuda --mesh-stride 2 \
  --mesh-depth-edge-threshold 0.12 --debug --work-dir workdir/mesh_right
```

组合编辑：

```bash
python app.py --video /data/ski.mp4 \
  --target-state examples/63_combined__depthmesh__zoom_in__subject_right.json \
  --depth-path workdir/mesh_right/reference_depth.npy \
  --mesh-device cuda --device cuda --debug --work-dir workdir/mesh_combined
```

这两次必须保持相同视频、seed和采样间隔，以确保选到相同参考帧。`--depth-path`会复用.npy旁的metric JSON，而非重新推理。

一次跑齐六个正式workflow并生成同深度A/B：

```bash
python tools/run_mesh_acceptance.py --video /data/ski.mp4 --output-dir workdir/mesh_server_acceptance
```

该脚本运行none、rotation、旧point、mesh右移、mesh前移、mesh+viewport+人物，全部调用app.py。首次mesh使用真实Depth Pro，后两例复用同一参考帧的metric深度；脚本不会冒充CPU执行GPU验收。

## 同深度 point vs mesh A/B

```bash
python tools/compare_point_mesh.py \
  --result workdir/mesh_right/result.json \
  --output-dir workdir/ab_same_depth --mesh-device cuda --mesh-stride 2
```

输出point_vs_mesh.jpg：Point | Mesh | 两者viewport之前的coverage mask。另有comparison.json和两套完整debug产物。两次使用同一reference/depth/K/运动/viewport/subject目标，隔离representation差别。单独比较旧Depth Anything+point与Depth Pro+mesh，会同时改变模型与representation，不应把全部差异归因于mesh。

result.json引用当前主机的绝对路径；搬迁结果目录后需修正这些路径，或直接在服务器重新跑workflow。工具未接入Agent正式输出逻辑。

## Debug字段与文件

- depth_metric.npy：仅metric输入保存原始米制Z；relative输入不伪装成metric文件。
- reference_depth.npy / reference_depth.json：后续可复用的深度与完整sidecar。
- reference_depth_visualization.png：可视化专用，不可用作深度输入。
- depth_edge_mask.png：原分辨率相邻深度断层。
- mesh_valid_vertex_mask.png、mesh_face_mask.png：采样网格有效性与两面标志。
- viewpoint_mesh_render.jpg、viewpoint_valid_mask.png：viewport前mesh结果及真实coverage。
- background_canvas.jpg、target_sketch.jpg：后续viewport与人物合成。
- mesh_metadata.json、result.json：backend、depth来源、metric/focal/intrinsics_source、顶点/候选面/保留面/各类裁面数、stride、threshold、coverage、角度平移、device/rasterizer、构建/光栅化/总耗时。

裁面计数按invalid→depth edge→subject region排他计数，便于求和；另记camera-plane裁面。Depth Pro metadata单独记录模型加载、推理与后处理时间；depth node还记录整个estimator耗时。Mesh计时不包含深度估计（独立记录）与最终viewport/人物合成。

## CPU实际验证与待GPU验证

本机：192项通过、11项GPU测试跳过（跳过原因显式显示）。

已完成：
- metric/relative存储、sidecar复用与真实HF后处理器测试；Depth Pro模型输出为synthetic。
- constant plane identity、连续平面平移、Z=1/5断层不连面、边界保留空洞、薄边界guard、人物mask裁面。
- 运动方向与旧point解析投影一致、metric scene尺度换算不改变Agent参数语义。
- 三个mesh完整workflow用例：右移、前移、组合，使用明确标注的CPU小网格测试oracle。
- 无依赖时在加载Depth Pro之前明确报错，旧模式保持运行。
- 滑雪视频真实app.py重跑none/rotation/depth_3d，与之前保存的三份JPEG逐字节一致。
- 真实滑雪参考帧（旧Depth Anything深度）CPU拓扑检查：stride2，519901顶点、1036800候选面；threshold .12保留954470面、裁掉82330个depth-edge面，构建约0.28秒。此结果不是Depth Pro或GPU效果验证。

本机记录：`workdir/depth_mesh_cpu_acceptance/`，含ski_topology_report.json、ski_depth_edges_overlay.jpg、真实旧workflow重跑以及mesh_expected_dependency_error/result.json。

必须等GPU：
- Depth Pro真实权重的metric深度/focal质量、显存与推理耗时。
- PyTorch3D安装兼容性、实际NDC方向、透视插值、遮挡、共享边coverage。
- 11项GPU测试（identity两个stride、flat translation、depth discontinuity、Z-buffer/透视颜色、6个方向测试）。
- 六个正式workflow的真实GPU运行，特别是组合编辑。
- 同深度point_vs_mesh.jpg和真实场景的连续面小孔/边界拉丝改进。

因此当前不声称Depth Pro+mesh已经在照片上优于baseline。

## 参数与限制

首轮建议stride2、threshold .12，yaw/pitch±3°起步，translation±.02～.05，roll±2°。之后按用户建议范围逐步增加，GPU高质量对照可用stride1。阈值越小，断层切得越保守，也可能破坏斜坡连续面；full-resolution edge guard有意扩大断层附近留白。

这仍是单帧2.5D：隐藏表面不存在、估计depth/focal会错、薄结构可能被切掉、vertex RGB在stride>1时丢高频细节、人物独立合成不生成新姿态。严禁以crop/zoom/大范围replicate来隐藏这些问题。

## 新增/修改文件清单

新增：
- photography_viewpoint_agent/depth_estimation/storage.py、depth_pro.py、factory.py
- photography_viewpoint_agent/renderer/mesh_warp.py
- requirements-mesh.txt
- examples/60_depthmesh__identity.json、61_depthmesh__translate_right_small.json、62_depthmesh__translate_forward_small.json、63_combined__depthmesh__zoom_in__subject_right.json
- tools/compare_point_mesh.py、tools/run_mesh_acceptance.py
- tests/test_depth_mesh.py、tests/test_mesh_gpu.py、tests/helpers/mesh_oracle.py
- docs/depth_mesh.md

修改：
- schemas/target.py、schemas/depth.py、config/settings.py
- depth_estimation/monocular.py（旧导入路径重导出storage函数）
- nodes/estimate_reference_depth.py、nodes/render_sketch.py、graph/workflow.py
- renderer/target_sketch.py、renderer/point_cloud_warp.py、renderer/viewpoint_warp.py
- tools/sketch_validator.py、photography_viewpoint_agent/app.py
- README.md、examples/README.md
