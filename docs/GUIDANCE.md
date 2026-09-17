# Photography Guidance Agent

Guidance Agent 负责 Target Follower / Controller：PhotoAgent 决定“应该拍成什么样”，本项目决定“用户现在应该怎么做”。

**Breaking change: crop-based composition generation has been replaced by PhotoAgent Target Sketch.**
旧随机目标 demo、preprocessing、CropBox、locate_crop、裁剪图输出已删除，不提供兼容接口。
`target_frame` 改为 `reference_frame`；`composition` 改为 `target_sketch`，并必须携带 `target_state` 与 `render_meta`。

## 安装

本仓库包含上下游，使用 Python 3.11+：

```powershell
python -m pip install -e ".[web,test]"
python -m pytest -q
```

## 输入与输出

输入：`video_path`、`current_frame`、`reference_frame`、`target_sketch`、完整 JSON `target_state`、`render_meta`。
图片与视频字段为本地路径。可选 `session_id`、正整数 `step_id`、`video_url`，另需 `session_directory` 指定归档目录。
下游 dataclass/TypedDict 保持独立，不导入上游 AgentState 或 Pydantic 模型。新模式的扩面、padding、人物 bbox、rotation、depth_3d、depth_mesh 元数据原样传递。

```python
import json
from video_guide.core import build_guide_graph

inputs = {
    "video_path": "video.avi",
    "current_frame": "current.png",
    "reference_frame": "reference.png",
    "target_sketch": "target_sketch.png",
    "target_state": json.load(open("target_state.json", encoding="utf-8")),
    "render_meta": json.load(open("render_meta.json", encoding="utf-8")),
    "session_directory": "outputs/session-1",
    "session_id": "session-1", "step_id": 1,
}
graph = build_guide_graph()  # LocalBackend; offline
output = graph.invoke(inputs)
print(output["result"])
# Same locked target; only current frame and step change:
output = graph.invoke({**inputs, "current_frame": "new_current.png", "step_id": 2})
```

结果：`phase`（navigation/composition/reached/uncertain）、`reference_reached`、`target_reached`、`actions`、`guidance`、`confidence`、`warnings`、`evidence`。
每个 action 包含 actor（camera/subject/lens）、action、可选 magnitude 和 reason。方向以摄影师画面为准；不输出后处理裁剪参数。

- navigation：两项 reached=false，只允许 camera 动作。
- composition：reference_reached=true，target_reached=false，可输出相机、人物、镜头动作。
- reached：两项 reached=true，actions=[]。
- uncertain：证据不足，target_reached=false，actions=[]；reference_reached 表示参考视角是否已确认。

## 图与归档

```text
START -> validate_input -> prepare_inputs -> analyze_alignment -> persist_result -> END
```

每个 step 正常仅一次 backend 调用。HTTP 429/500/502/503/504、网络错误和超时最多尝试两次；协议错误不重试、不静默回退。
保留 checkpointer、输入/结果校验、临时目录与原子发布，详见 [LangGraph 集成](LANGGRAPH_INTEGRATION.md)。

```text
session/
  target/
    reference_frame.png
    target_sketch.png
    target_state.json
    render_meta.json
    manifest.json
  guidance/
    step_000001/
      current_frame.png
      execution.json
      manifest.json
      result.json
    step_000002/...
```

扩展名沿用原图。目标只复制一次，内容校验防止同一 session 改目标；每步只复制当前图，不复制视频、参考图或目标图。
视频使用稳定的源文件绝对路径，调用方须保持文件可读。Web 上传的视频在 session/source 保存一次。
省略 step_id 自动分配；显式重复编号拒绝覆盖。失败保留 `.pending-step_*` 供检查点恢复，自动编号会跳过占用编号。
同一 session 内 generation 所需原始帧/debug 文件可独立保存在 generation/。

## CLI / Web

```powershell
python -m video_guide guide --video-path video.avi --current-frame current.png --reference-frame reference.png --target-sketch target.png --target-state target_state.json --render-meta render_meta.json --session-directory outputs/session-1 --step-id 1
python -m uvicorn video_guide.web:app --host 127.0.0.1 --port 8000
```

CLI 用 `--backend vlm --config config/vlm.json` 启用 VLM。Web 上传视频、三张图，并填写 TargetState/RenderMeta JSON；每次 Web 提交创建新 session 的第一步，连续步骤由 Python API/CLI 调用。
`POST /api/analyze` multipart 字段：video、current_frame、reference_frame、target_sketch、target_state、render_meta、backend、可选 video_url。
结果图通过 `GET /api/runs/{session_id}/{filename:path}` 获取，路径限制在当前 session 内。Web 仅作本机演示，无用户鉴权。

## Backend 协议与限制

```python
analyze(current, reference, target_sketch, video, target_state, render_meta, video_url=None) -> GuideResult
```

三张图为 RGB PIL Image，video 为原视频 Path，结构数据为 JSON 字典。VLM 接收三张有标签的图、全部元数据和完整视频/URL；保持配置读取、FPS、超时与传输限制，见 [Qwen 配置](QWEN_SETUP.md)。
先判断参考视角，再判断目标构图；合法的扩面和人物/深度变换不要求可在参考图内定位。相机为实现目标而做出的必要平移不应触发反向导航。

LocalBackend 仅供保守的离线验证：有图像匹配证据时可给朝向动作；简单整图相似可判 reached；复杂构图、人物重排、扩面和深度目标返回 uncertain，交给 VLM 做语义判断。不会根据目标相对参考的编辑量，假装知道当前剩余的三维动作。
真实 VLM 的指引质量尚需真实视频评测；单目输入不保证绝对距离或路径恢复。


## GLM 实际通路测试（2026-09-17）

滑雪视频已经通过 PhotoAgent → TargetPackage → 真实 glm-5.3-flash → GuideResult 解析/落盘。
此服务的完整视频请求约 44.8 MB，发生连接重置；成功测试显式使用 images 模式，发送 Current/Reference/Target Sketch 和完整 TargetState/RenderMeta，不发送整段视频。
模型要求思考，使用 reasoning_effort="low"，不发送 enable_thinking=false。

```python
import os
from video_guide.core.backends import VLMBackend
backend = VLMBackend(
    endpoint="https://st8tp3ajl0df3n8b8l8qu.apigateway-cn-beijing.volceapi.com/v1/chat/completions",
    model="glm-5.3-flash",
    api_key=os.environ["VLM_API_KEY"],
    input_mode="images",
    reasoning_effort="low",
)
# build_integrated_workflow(config, backend=backend)
```

input_mode 默认为 video；images 是显式选择，不自动降级。reasoning_effort 默认为 None，保持既有请求行为。以上新参数通过 Python 构造器指定。
本次仅验证通路，不评估动作精度。Key 未写入代码、配置或结果文件。
