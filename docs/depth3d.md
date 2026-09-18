# depth_3d：实验旋转 backend 与低层点云实现

## 设计与接入

正式 viewpoint.mode 仅 none/rotation；选择 AgentConfig.viewpoint_backend=depth_3d 且角度非零时使用点云旋转，camera translation 固定为0。语义规范见 [TRANSFORM_SEMANTICS](TRANSFORM_SEMANTICS.md)。

```text
load_video → extract_frames → select_reference_frame + get_current_frame
  → composition_planner
  → [subject=reposition 时 detect_reference_subject]
  → [非零旋转 AND backend=depth_3d 时 estimate_reference_depth]
  → render_target_sketch → validate_sketch
```

路由读取 planner 最终输出。零角度或 homography backend 不调用深度估计，构建 graph 也不加载模型权重。原四参数 renderer 注入接口在旧模式下照常调用，depth_3d 额外传入 keyword reference_depth。

AgentState.reference_depth 为 DepthObservation：depth_path、width、height、Z-depth 表示、median_one 归一化约定及估计来源/统计。估计器通过 build_workflow(depth_estimator=...) 可替换。

### 渲染路径（backend 来自配置）

- none：原有处理。
- rotation + homography：保留 Homography；零角度跳过所有 viewpoint renderer。
- depth_3d + follow_reference：整图 RGB/Z → 点云渲染 → 原有 reference_viewport。
- depth_3d + reposition：原始图像分割人物 → 修补背景 RGB 和对应深度 → 只对背景进行点云渲染 → reference_viewport → 按原逻辑 contain 缩放并合成人物。

背景深度使用与 RGB 相同的7×7膨胀 mask，并通过 OpenCV Navier–Stokes float inpaint 修补；只改 masked 区域，保留原归一化尺度。人物层保留原始姿态和视角，不当作真实新视角人物。

深度 backend 的独立人物编辑不使用 no-op 优化。其 natural_subject_bbox 留空：修补后的背景已经没有原人物可供投影；source_subject_bbox 仍用于验证人物 contain 几何。follow_reference 有 observation 时可记录源 bbox 内点的投影范围，但不是精确遮挡后分割框。

## 深度模型与尺度

默认模型：[Depth Anything V2 Small HF](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)，Small 版本 Apache-2.0；约24.8M参数。通过 Transformers 的 AutoImageProcessor / AutoModelForDepthEstimation 加载，CPU 已真实运行。不会执行仓库自定义 remote code。

本 adapter 明确接受 Depth Anything 的 relative inverse-depth 权重，不把任意模型的输出都当作 Z。其他模型应实现替代 DepthEstimator。

转换约定：

```text
p2, p98 = raw inverse-depth 的 2% / 98% 分位
q = clip((raw - p2) / max(p98 - p2, epsilon), 0, 1)
inverse = 0.1 + 0.9*q
Z = 1 / inverse
Z = Z / median(Z)
```

这是解决相对深度仿射歧义的一种固定启发式，不恢复真实距离。仅低层研究中的 translation=0.03 表示源相机位移等于归一化场景中位 Z 的3%，不是3厘米，也不是图宽的3%。各帧的尺度仍可能变化。

预计算输入：`--depth-path depth.npy`，要求二维、与选中的参考帧尺寸完全一致、正值表示更远的相机 Z。**不能直接传原始 inverse depth 或上色的 depth PNG。** 输入会以有效值中位数归一化；NaN/Inf/非正值排除；全无效/尺寸错误明确失败。更换视频/帧后不能复用旧深度；当前仅能检查尺寸，不能证明文件对应同一场景。

## 几何与方向（保留低层 R+t；正式 C 恒为0）

x向右、y向下、z向前；列向量，像素中心索引为整数。

```text
f = (W/2)/tan(FOVx/2)
K = [[f,0,W/2],[0,f,H/2],[0,0,1]]
P = [(u-W/2)*Z/f, (v-H/2)*Z/f, Z]
R = Rz(-roll) @ Rx(-pitch) @ Ry(-yaw)
C = [translation_x, -translation_y, translation_z]
P_target = R @ (P - C)
u_target = f*X_target/Z_target + W/2
v_target = f*Y_target/Z_target + H/2
```

平移方向定义在**源相机坐标系**，不是先旋转后的局部轴。等价外参 t = -R@C。旋转顺序及符号复用 rotation backend。

| 正参数 | 相机动作 | 画面趋势 |
| --- | --- | --- |
| yaw | 向右转 | 场景向左 |
| pitch | 抬头 | 场景向下 |
| roll | 沿前方看顺时针转 | 场景逆时针 |
| translation_x | 向右移动 | 场景向左，近处更多 |
| translation_y | 向上移动 | 场景向下，近处更多 |
| translation_z | 向前移动 | 物体向画面中心外扩、近景放大更多 |

负号反向。纯旋转时，点投影在光栅化之前与 H=KRK^-1 一致，与深度值无关。

### Splat / z-buffer

默认 splat_radius=1，即每点覆盖3×3邻域（可设置1～3，对应3×3～7×7）。先按目标相机 Z 选最近表面，同深度取距离像素中心最近的样本，再按源像素索引稳定打破平局。不是仅round到单像素。重复运行确定性一致。

无相机运动时保留有效源像素，不运行 splat，避免 identity 时邻近深度侵蚀纹理。Z<=1e-6 的点剔除，全部移出画布时明确失败。

- constant：未覆盖区采用配置 fill_color，默认灰色。
- replicate：将未覆盖区扩展为最近已覆盖像素颜色，可能有明显拉伸；coverage仍记录填充前的原始覆盖率。

不做 safe crop / auto zoom / 补裁切 / 生成式修补。旋转后画布仍是原始尺寸，后续只执行用户显式 reference_viewport（保留原有长宽比适配）。

## 安装与运行

```powershell
python -m pip install -r requirements.txt
```

本机通过 `.venv311` 运行；其他机器替换 Python 路径。首次模型推理需要下载权重，也可通过 --depth-model 指向本地目录。

```powershell
.\.venv311\Scripts\python app.py `
  --video D:/Data/videoagent/test_data/PhotoAgent/mixkit-children-skiing-on-the-plain-of-a-pine-forest-3349-full-hd.mp4 `
  --target-state examples/viewpoint_yaw_right.json --viewpoint-backend depth_3d `
  --work-dir workdir/depth3d_right_new `
  --depth-model workdir/depth_models/depth-anything-v2-small `
  --depth-device cpu --device cpu --splat-radius 1 --debug
```

服务器可指定 --depth-device cuda。--device 控制已有 YOLO，二者独立。--depth-debug 仅开启深度可视化，--debug 开启全部图层。work-dir 必须为空。

已在本机生成的真实深度可复用：

```powershell
.\.venv311\Scripts\python app.py `
  --video D:/Data/videoagent/test_data/PhotoAgent/mixkit-children-skiing-on-the-plain-of-a-pine-forest-3349-full-hd.mp4 `
  --target-state examples/framing_plus_subject_plus_yaw.json --viewpoint-backend depth_3d `
  --depth-path workdir/depth3d_acceptance/identity/reference_depth.npy `
  --work-dir workdir/depth3d_combined_new --debug
```

以上固定默认 seed42 / interval30，参考帧为第30帧。`--depth-path` 优先于模型配置，仍经过正式 depth node 读取、验证和存储。

## 示例索引

正式案例使用 viewpoint_yaw_right.json、framing_plus_subject_plus_yaw.json；同一 TargetState 可选择不同 renderer 配置。旧50–55示例迁至 [research only](../experiments/camera_translation/README.md)，不得传给 app.py。

## Debug 输出

每次完整运行输出 result.json、target_sketch.jpg、reference_frame.jpg 和已有视频采样产物。depth_3d 另有：

- reference_depth.npy：归一化正Z深度。
- reference_depth_visualization.png：debug/depth-debug启用时；按逆深度着色，暖色偏近、冷色偏远，仅供显示。
- viewpoint_warped_background.jpg：viewport之前的深度变换结果。
- viewpoint_valid_mask.png：白色表示真实投影覆盖，填充前。
- background_depth.npy：渲染使用的Z，独立人物模式下已修补。
- background_without_subject.jpg、background_canvas.jpg、subject_layer.png、placed_subject.png：对应图层（人物文件只在独立编辑时输出）。

后三组图像通过 --debug 保存。result.json 的 render_meta.viewpoint_warp 包含参数、K、R、源坐标相机中心、外参t、coverage、depth_stats、kernel和背景深度是否修补；reference_depth.metadata记录来源、模型、转换参数。旧padding仅描述viewport矩形边缘，不能代替点云coverage。

## 历史完整 workflow 验收（旧语义，仅保留实验记录）

以下数字来自重构之前，含平移的旧 workflow 已退出正式 API。当前验证以 TRANSFORM_REFACTOR_DELIVERY.md 为准。

路径：`D:/Project/PhotoAgent/workdir/depth3d_acceptance/`。所有以下用例均通过 app.py 处理完整视频并经过图节点，不是独立renderer demo。参考帧1920×1080、输出1280×720、CPU。

| 用例目录 | depth来源 | coverage（viewport前） | 耗时约 |
| --- | --- | --- | --- |
| none | 跳过 | — | 2.8秒 |
| rotation | 跳过 | 89.4% | 2.7秒 |
| identity | 真实Small模型推理 | 100% | 9.4秒 |
| right | 同一参考帧的预计算真实深度 | 95.3% | 7.0秒 |
| forward | 同上 | 99.8% | 6.6秒 |
| backward | 同上 | 89.1% | 7.2秒 |
| up | 同上 | 93.8% | 6.6秒 |
| combined | 同上，人物检测为真实CPU YOLO | 85.0% | 11.1秒 |

none / rotation 故意提供不存在的depth model，均成功，确认没有隐式推理。identity与none的最终JPEG **字节完全一致**。

验收对照图：workflow_comparison.jpg。记录：acceptance_summary.json、parallax_measurements.json、每个子目录result.json及运行日志。

右移0.03的解析投影检查：源图人物mask内模型Z中位数约0.362，远处树林ROI的Z约1.691；预计向左位移中位数分别约137.8与29.5源图像素。这个量来自模型深度的投影，不是对真实相机运动的实测。可视化符合近景移动更多、前移放大、后移缩小的趋势。

全部8例workflow validation通过；PASS仅表示接口、几何参数、输出尺寸和人物目标位置等通过，**不代表深度准确或图像逼真**。钢架、人物边缘有点云空洞、轮廓拉扯，雪地深度断层会留下条纹。组合例仍有原滑雪板残留、人物分割白边，这是已有mask不包含滑雪板与修补近似的限制。

## 低层研究测试与建议范围（非正式 composition API）

```powershell
.\.venv311\Scripts\python -m pytest tests experiments/fit_target_state/tests -q
```

当前170项测试通过。新增：schema/平移约束、median归一化、identity像素一致、确定性、近远视差、上下/前后方向、纯旋转与H一致、近表面z-buffer优先、3×3覆盖、replicate、背面剔除、背景深度修补；完整图覆盖3 backend × 2主体模式、只在depth模式调用估计器、预计算CLI及无效depth错误传播。

建议从 yaw/pitch ±3°、roll ±2°、translation各轴 ±0.005～0.02 开始。示例0.03便于观察，但本图近景位移已明显。参数硬限制角度±15°、平移±0.2并非推荐范围；组合运动、极近物体、宽FoV更易出空洞。没有对其他场景的稳定性作保证。

## 历史实现文件清单（旧示例已迁移研究目录）

新增：
- schemas/depth.py
- depth_estimation/__init__.py、base.py、monocular.py
- nodes/estimate_reference_depth.py
- renderer/point_cloud_warp.py
- tests/test_depth3d.py
- examples/50～55上述六个JSON
- docs/depth3d.md

修改（包内路径 photography_viewpoint_agent/）：
- schemas/target.py、schemas/state.py
- config/settings.py、app.py
- graph/workflow.py、nodes/render_sketch.py
- renderer/base.py、renderer/target_sketch.py
- tools/sketch_validator.py（兼容旋转/深度几何的校验，修复旧rotation+reposition误判）
- requirements.txt、examples/README.md、README.md
