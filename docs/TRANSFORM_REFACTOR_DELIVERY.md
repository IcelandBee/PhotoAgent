# 小范围职责收紧重构交付

## 1. 修改文件

完整清单见本文末尾。point_cloud_warp.py 与 mesh_warp.py 和重构前逐字节相同；没有重写 LangGraph 或优化几何算法。

## 2. 新 TargetState schema

```json
{"subject":{"mode":"reposition","bbox":[0.65,0.25,0.85,0.85]},"framing":{"reference_viewport":[0.1,0,1.1,1]},"viewpoint":{"mode":"rotation","yaw_deg":5,"pitch_deg":0,"roll_deg":0}}
```

viewpoint 只有 mode=none|rotation、yaw_deg、pitch_deg、roll_deg；省略时为 none/零角度。subject/framing 规则保持原样。详细约束见 [语义文档](TRANSFORM_SEMANTICS.md)。

## 3–4. 职责与执行顺序

viewpoint=固定光心朝向；framing=整幅画面二维取景/缩放/裁切/扩面；subject=人物相对背景在最终画布中的位置和尺度。

执行顺序：**Viewpoint Rotation → reference_viewport → Subject Layout**。人物独立编辑保留提取/背景修补/最后合成，不在最后再次旋转人物 bbox。

## 5–6. Viewpoint 与 depth routing

AgentConfig.viewpoint_backend 默认 homography；可选 depth_3d/depth_mesh。角度全零时跳过所有 viewpoint warp、depth estimator 和 GPU preflight。非零旋转时，homography 不需要深度，两个 depth backend 才进入深度节点，并且 translation 恒为0。FoV、border_mode 同样归配置。

## 7–8. 平移隔离与保留

正式输入拒绝 translation_x/y/z，包括值0；拒绝旧 depth mode、FoV、border。Planner 输出验证、renderer 入口、集成输入/handoff、Guidance graph/backend 形成边界校验；无静默转换。低层 CameraWarpParameters 承接旧数学参数，正式 rotation_parameters adapter 显式注入零平移。

底层 point/mesh 的 R+t 几何与平移测试保留。十个旧例迁至 experiments/camera_translation/examples，独立 runner 直接调用低层渲染，不运行旧例的 framing/subject。这是 research only，不新增正式 translation intent。

## 9. Guidance

Prompt 区分 desired 2D framing / explicit orientation / desired subject layout；强调 viewport 正向 shift 的采样含义、最终 bbox 和执行顺序，禁止只凭 viewport 推导 move_right 等物理平移。导航阶段有独立图像证据的实际移动动作仍保留。原 translation_x/y/z 语义提示已移除。

## 10. Examples

新增 framing_shift_right、framing_zoom_in、viewpoint_yaw_right、viewpoint_pitch_up、framing_plus_yaw、subject_move_right、framing_plus_subject_plus_yaw 七个案例。原40/41案例移除 renderer 配置字段。根目录21个 JSON 全部通过新 schema。

## 11–12. 测试与实际运行

- 新增 tests/test_transform_semantics.py：零旋转不调用warper、subject独立背景不动、yaw像素对照与错误顺序对照、最终bbox、组合变换、三backend路由、旧字段拒绝、metadata一致性、低层研究平移保留。
- Guidance tests 新增 graph/backend 旧输入提前拒绝；integration tests 新增 handoff 拒绝；旧点云/mesh/GPU测试改用低层参数，保留原平移能力覆盖。
- 完整 `python -m pytest -q --tb=short`：**283 passed, 11 skipped, 1 warning（12.66s）**。11项为CUDA/PyTorch3D GPU测试；warning为已有Starlette/AnyIO弃用提示。
- 独立 `experiments/fit_target_state/tests`：**17 passed**。
- 本地滑雪视频实际app.py运行3例：framing+mesh配置（实际backend=none、无depth），yaw+homography（无depth），yaw+point（复用已计算参考深度）。三例validation均通过，camera_translation均为[0,0,0]。产物在本机 workdir/transform_semantics_acceptance/，不提交媒体或模型。
- 本轮没有重新运行真实VLM请求或GPU mesh，也不声称其图像质量通过。

## 13. 全仓搜索审查

检查 photography_viewpoint_agent、video_guide、tools、experiments 中全部 TargetState producer/consumer、viewpoint.mode 与 translation_x/y/z 引用：

- 公共 schema 无平移字段；Manual Planner 输出由 validate_plan 重新验证；CLI和集成输入均校验。
- workflow/node 的 depth 分支由配置与非零角度共同决定。
- 正式 renderer 中出现的 translation 仅为已隔离低层参数的尺度诊断；adapter 只能构造零值。
- Guidance 无 viewport→translation 计算代码，且拒绝旧字段；VLM prompt 明确禁止这种推断，但这不构成对任意模型自然语言输出的绝对保证。
- fit_target_state 仍只拟合二维viewport/subject；A/B与GPU验收工具改用相同rotation intent和不同backend配置。
- 非零平移保留位置仅低层R+t实现、研究入口/归档案例和对应能力测试；历史文档标为历史记录。
- 未发现正式 TargetState / Planner / Guidance 中可将 framing shift 映射为 camera translation 的执行路径。

## 完整变更路径

```text
M	README.md
M	docs/GUIDANCE.md
M	docs/INTEGRATION_DELIVERY.md
M	docs/TARGET_GENERATOR.md
A	docs/TRANSFORM_REFACTOR_DELIVERY.md
A	docs/TRANSFORM_SEMANTICS.md
M	docs/depth3d.md
M	docs/depth_mesh.md
M	examples/40_viewpoint_only__yaw_right_5deg.json
M	examples/41_combined__rotation__zoom_in__subject_right.json
M	examples/README.md
A	examples/framing_plus_subject_plus_yaw.json
A	examples/framing_plus_yaw.json
A	examples/framing_shift_right.json
A	examples/framing_zoom_in.json
A	examples/subject_move_right.json
A	examples/viewpoint_pitch_up.json
A	examples/viewpoint_yaw_right.json
A	experiments/camera_translation/README.md
R100	examples/50_depth3d__identity.json	experiments/camera_translation/examples/50_depth3d__identity.json
R100	examples/51_depth3d__translate_right_small.json	experiments/camera_translation/examples/51_depth3d__translate_right_small.json
R100	examples/52_depth3d__translate_forward_small.json	experiments/camera_translation/examples/52_depth3d__translate_forward_small.json
R100	examples/53_depth3d__translate_backward_small.json	experiments/camera_translation/examples/53_depth3d__translate_backward_small.json
R100	examples/54_combined__depth3d__zoom_in__subject_right.json	experiments/camera_translation/examples/54_combined__depth3d__zoom_in__subject_right.json
R100	examples/55_depth3d__translate_up_small.json	experiments/camera_translation/examples/55_depth3d__translate_up_small.json
R100	examples/60_depthmesh__identity.json	experiments/camera_translation/examples/60_depthmesh__identity.json
R100	examples/61_depthmesh__translate_right_small.json	experiments/camera_translation/examples/61_depthmesh__translate_right_small.json
R100	examples/62_depthmesh__translate_forward_small.json	experiments/camera_translation/examples/62_depthmesh__translate_forward_small.json
R100	examples/63_combined__depthmesh__zoom_in__subject_right.json	experiments/camera_translation/examples/63_combined__depthmesh__zoom_in__subject_right.json
A	experiments/camera_translation/run.py
M	photography_viewpoint_agent/app.py
M	photography_viewpoint_agent/config/settings.py
M	photography_viewpoint_agent/graph/integrated_workflow.py
M	photography_viewpoint_agent/graph/workflow.py
M	photography_viewpoint_agent/integration/guidance_adapter.py
M	photography_viewpoint_agent/nodes/estimate_reference_depth.py
M	photography_viewpoint_agent/nodes/render_sketch.py
M	photography_viewpoint_agent/planners/base.py
A	photography_viewpoint_agent/renderer/camera_parameters.py
M	photography_viewpoint_agent/renderer/target_sketch.py
M	photography_viewpoint_agent/renderer/viewpoint_warp.py
M	photography_viewpoint_agent/schemas/rendering.py
M	photography_viewpoint_agent/schemas/target.py
M	photography_viewpoint_agent/tools/sketch_validator.py
M	tests/guidance/test_config_video.py
M	tests/guidance/test_graph.py
M	tests/test_depth3d.py
M	tests/test_depth_mesh.py
M	tests/test_integration.py
M	tests/test_mesh_gpu.py
A	tests/test_transform_semantics.py
M	tests/test_viewpoint_warp.py
M	tools/compare_point_mesh.py
M	tools/run_mesh_acceptance.py
M	video_guide/core/backends.py
M	video_guide/core/models.py
```
