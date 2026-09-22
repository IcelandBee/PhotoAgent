# PhotoAgent：Depth + Point Cloud 目标生成与拍摄指导

所有 Target Sketch 统一使用：Source RGB → Depth → Colored Point Cloud → Camera R/t/K_target → Target Sketch。
人物固定在场景中，随场景几何自然投影。支持真实相机平移、旋转、焦距和取景范围变化。
零运动也经过点云投影。未知区域固定颜色留白，同时输出 valid mask。

## 安装与运行

Python 3.11+；首次模型推理需要下载 Depth Anything V2 Small，或用 --depth-model 指定已有本地模型目录。

```sh
python -m pip install -e ".[web,test]"
python app.py --video sample.mp4 --target-state examples/camera_up_pitch_down.json --work-dir workdir/run-001 --depth-device cpu --debug
python -m pytest tests experiments/depth_pointcloud_sketch/test_batch.py -q
```

--depth-device cuda 只加速深度推理；点云渲染当前使用 NumPy CPU。
正式 Renderer 默认是 depth-aware bilinear splatting，可用 --point-renderer nearest_z 显式切换到回归/调试基准。
27 图实验中，identity MAE 从 11.730 降至 0.0078，Renderer 中位耗时从 1.286s 降至 0.860s；
平均有效覆盖率从 90.32% 降至 89.46%。Gaussian 仅保留在研究实验中。
每次运行选择新的输出目录。默认输出 1280×720，可通过 --target-width/--target-height 修改。
--depth-path 可复用与选定参考帧严格匹配的正 Z 深度，不接受原始 inverse-depth。
可选 Depth Pro 只是深度估计器替换，不改变点云渲染通路。
requirements-lock.txt 保留历史 CPU 环境版本记录，不包括 Web extras；推荐按 requirements.txt 安装。

## 目标参数

```json
{
  "subject": {"mode": "follow_reference"},
  "framing": {"reference_viewport": [0, 0, 1, 1], "focal_scale": 0.9},
  "viewpoint": {
    "mode": "camera",
    "translation_y": 0.05,
    "pitch_deg": -8
  }
}
```

+x 向右，+y 向上，+z 前进；+yaw 向右转，+pitch 仰拍，+roll 顺时针。
平移单位是场景中位深度的比例，不是米。参数默认 0，focal_scale 默认 1。
旧 none / rotation 意图仍可输入，但同样必须估计深度并投影点云。
reposition、人物 bbox、Homography/Mesh 后端与 replicate 填充已移除，非法输入明确报错。
reference_viewport 在目标 K 内完成取景映射；它本身不自动推断机位平移。
需要近景视差时显式设置 translation_z，例如 camera_forward_zoom_in.json。

## 正式 workflow 和 Guidance

```text
Video → Extract Frames → Reference + Current → Planner
      → Estimate Reference Depth → Point Cloud Target Sketch → Validate
      → TargetPackage → Guidance → Persist Result
```

Planner 当前仍使用人工 TargetState，可通过接口注入其他 planner。
参考帧深度每次目标生成只计算一次；后续 Guidance steps 复用已锁定的目标，不重新生成。
缺少深度或推理失败会返回错误，不进行其他渲染回退。

```python
from photography_viewpoint_agent.config.settings import AgentConfig
from photography_viewpoint_agent.graph.integrated_workflow import build_integrated_workflow

workflow = build_integrated_workflow(AgentConfig(depth_device="cpu"))
result = workflow.invoke({
    "video_path": "sample.mp4",
    "manual_target_state": {
        "viewpoint": {"mode": "camera", "translation_y": 0.05, "pitch_deg": -8}
    },
    "session_directory": "workdir/session-001",
    "session_id": "session-001", "step_id": 1,
})
print(result["guidance"])
```

默认 LocalBackend 保守返回 uncertain，不把合成草图匹配当成物理机位已到达。
实际语义指导可注入 VLMBackend；不会自动发起付费模型调用。Prompt 理解相机平移和 focal_scale，
且 scene-fixed 流程拒绝独立人物动作。Guidance 中的图像匹配只用于参考帧定位，不参与 Target Sketch 渲染。

离线完整链路 demo（明确使用合成平面深度，不代表真实模型推理）：

```sh
python tools/run_integrated_demo.py --output workdir/offline-demo
```

## 批量单图实验与 Viewer

```sh
python experiments/depth_pointcloud_sketch/run_batch.py --input-dir /path/to/images --output-dir workdir/batch-001 --device cpu --save-depth --save-debug
```

保留 10 个清晰 preset；每张图只估计一次深度、构建一次基础点云。
正式渲染使用 camera_rendering/depth_renderer.py；这个历史批量实验显式保留 nearest-Z 输出以便对照。

Viewer 已独立到 [IcelandBee/viewer](https://github.com/IcelandBee/viewer)。
本机代码目录 D:/Project/viewer；数据准备命令：

```sh
python /path/to/viewer/tools/prepare_viewer_data.py --experiment-dir /path/to/batch-001 --output-dir /path/to/viewer --overwrite
```

## 目录

- photography_viewpoint_agent/：目标 schema、视频处理、深度、统一点云和集成 workflow。
- video_guide/：Guidance graph、VLM/local backend、CLI/Web。
- experiments/depth_pointcloud_sketch/：原始相机渲染批量实验；experiments/renderer_interpolation_ablation/：Renderer 研究与回归。
- examples/：当前 schema 示例；tools/：完整链路 demo 和迁移验收。
- tests/：点云几何、workflow、Guidance、交接与服务测试。
- workdir/：模型缓存、保留的点云基准结果和新运行产物，Git 忽略。
- docs/POINTCLOUD_PIPELINE.md：实现、参数、迁移和局限。

旧代码、实验和历史产物移入仓库外可恢复归档，详见 [清理记录](docs/CLEANUP.md)。
[Guidance](docs/GUIDANCE.md) · [Graph/Checkpoint](docs/LANGGRAPH_INTEGRATION.md) · [VLM 配置](docs/QWEN_SETUP.md)
