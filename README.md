# Photography Agent：目标生成与拍摄指导

本仓库包含完整的 PhotoAgent 上游、Guidance 下游、集成 Graph、CLI/Web、示例及测试。
[IcelandBee/PhotoAgent](https://github.com/IcelandBee/PhotoAgent) 与 [roycyh/video-stream-based-photography-recomendation](https://github.com/roycyh/video-stream-based-photography-recomendation) 发布相同源码；克隆任意一个仓库即可安装和运行，无需再安装另一个仓库。

PhotoAgent 决定“应该拍成什么样”；Guidance 决定“用户现在应该怎么做”。两个模块保持明确的 schema 和职责边界，共用一个 Python 项目。

## 安装（Python 3.11+）

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[web,test]"
.\.venv\Scripts\python -m pytest -q
```

macOS/Linux 使用 `.venv/bin/python`。一次安装提供 `photography_viewpoint_agent` 和 `video_guide` 两个 import 包，以及 `photo-agent` / `video-guide` 两个命令。
从旧双仓环境迁移时，先卸载旧的 `video-streaming-recommendation` distribution，避免 editable 导入指向另一个目录。统一 distribution 名称为 photography-viewpoint-agent。

基础依赖沿用 requirements.txt；Web/test 通过 extras 安装。requirements-lock.txt 是原 CPU 核心环境快照，不包含新增 Web extras；GPU mesh 依赖见 requirements-mesh.txt 与 [说明](docs/depth_mesh.md)。

## 离线端到端 Demo

```powershell
python tools/run_integrated_demo.py --output workdir/integrated-demo
```

自动生成小视频，执行真实 Target Generation → Guidance，保存 integrated_result.json。无需 API Key、模型下载或 GPU；再次运行使用新的输出目录。

## 实际视频集成

```python
from photography_viewpoint_agent.config.settings import AgentConfig
from photography_viewpoint_agent.graph.integrated_workflow import build_integrated_workflow

workflow = build_integrated_workflow(AgentConfig(target_width=1280, target_height=720))
result = workflow.invoke({
    "video_path": "sample.mp4",
    "manual_target_state": {
        "subject": {"mode": "follow_reference"},
        "framing": {"reference_viewport": [-0.1, -0.1, 1.1, 1.1]},
    },
    "session_directory": "workdir/session-001",
    "session_id": "session-001", "step_id": 1,
})
print(result["guidance"])
```

默认 LocalBackend 用于保守离线验证，复杂构图返回 uncertain。VLM 通过 `build_integrated_workflow(config, backend=VLMBackend(...))` 注入，支持显式 video/images 模式。GLM 实际请求例子见 [Guidance 文档](docs/GUIDANCE.md)。Key 通过环境变量传入，不提交到 Git。

## Graph / Handoff

```text
START -> PhotoAgent Graph -> TargetPackage -> Adapter -> Guidance Graph -> END
Guidance: validate_input -> prepare_inputs -> analyze_alignment -> persist_result
```

TargetPackage 包含 video_path、reference_frame、current_frame、target_state、target_sketch_path、render_meta。Guidance 使用路径和完整 JSON 元数据，输出 navigation/composition/reached/uncertain、reached 标志、camera/subject/lens 动作与文字指导。

**crop-based composition generation has been replaced by PhotoAgent Target Sketch.** 不保留 CropBox、random_crop、locate_crop 或下游目标裁剪流程。

## 后续帧复用目标

```python
from video_guide.core import build_guide_graph
from photography_viewpoint_agent.schemas.handoff import TargetPackage
from photography_viewpoint_agent.integration import to_guide_input

package = TargetPackage.model_validate(result["target_package"])
next_result = build_guide_graph().invoke(to_guide_input(
    package, "workdir/session-001", current_frame="new_current.png",
    session_id="session-001", step_id=2,
))
```

Target Generation 不随当前帧重复执行。generation/ 保存生成阶段产物；target/ 固定保存参考图、目标图和元数据；guidance/step_XXXXXX/ 只保存本步当前图与结果。视频保持原路径可读，不按 step 重复复制。重试/检查点见 [Graph 文档](docs/LANGGRAPH_INTEGRATION.md)。

## 目录与测试

```text
photography_viewpoint_agent/  上游与集成入口
video_guide/                  下游、CLI、Web
examples/                     TargetState 示例与视频生成器
tests/                        上游与跨模块集成测试
  guidance/                   下游测试（避免同名冲突）
experiments/                  视点与目标拟合实验
tools/                        Demo 与渲染验证工具
docs/                         两模块说明与实测记录
```

```powershell
python -m pytest -q
python -m pytest tests/guidance -q
python -m pytest tests/test_integration.py -q
python -m uvicorn video_guide.web:app --host 127.0.0.1 --port 8000
```

GPU 测试在缺少 CUDA/PyTorch3D 时明确 skip。Web 是本机演示服务，每次提交创建新 session；实时摄像头循环不在本版范围。

- [Target Generator 详细说明](docs/TARGET_GENERATOR.md)
- [Guidance 输入、输出、CLI、GLM 适配](docs/GUIDANCE.md)
- [VLM 配置](docs/QWEN_SETUP.md)
- [初次整合记录（历史）](docs/INTEGRATION_DELIVERY.md)
- [滑雪视频真实 VLM 测试](docs/GLM_LIVE_TEST.md)

## 编辑语义

正式 TargetState 的 viewpoint 仅表示 yaw/pitch/roll，framing 表示整图二维构图，subject 表示最终人物布局；顺序固定为旋转→取景→人物。默认 homography，depth backend 移至 AgentConfig，只有非零旋转时使用，正式平移恒为零。详见 [Transform 语义与迁移](docs/TRANSFORM_SEMANTICS.md)。
