# PhotoAgent / Guidance 集成交付记录

> 历史测试/交付记录。当前两个仓库均包含完整上下游源码，安装方式以根目录 README 为准；以下本地路径指向当时的测试产物。

## 1. Architecture

PhotoAgent 保持 Target Generator 职责；video_guide 为 Target Follower / Controller。两个 Python 包独立 editable 安装到 PhotoAgent 的 Python 3.11.16 环境，未合并源码或内部 State。
上游现有 graph/nodes/renderer 未修改。下游沿用 StateGraph、input/output schema、backend abstraction、瞬时错误重试、检查点恢复、结果校验与原子发布。
单步只调用一次 backend，由提示词和结构校验明确两层判断。导航阶段只输出 camera 动作，构图阶段允许 camera/subject/lens；证据不足返回 uncertain。

## 2. Modified Files

PhotoAgent：

- [README.md](D:/Project/PhotoAgent/README.md)

video_guide：

- [README.md](D:/Project/video-stream-based-photography-recomendation-main/README.md)
- [docs/LANGGRAPH_INTEGRATION.md](D:/Project/video-stream-based-photography-recomendation-main/docs/LANGGRAPH_INTEGRATION.md)
- [docs/QWEN_SETUP.md](D:/Project/video-stream-based-photography-recomendation-main/docs/QWEN_SETUP.md)
- [pyproject.toml](D:/Project/video-stream-based-photography-recomendation-main/pyproject.toml)
- [tests/test_config_video.py](D:/Project/video-stream-based-photography-recomendation-main/tests/test_config_video.py)
- [tests/test_graph.py](D:/Project/video-stream-based-photography-recomendation-main/tests/test_graph.py)
- [tests/test_web.py](D:/Project/video-stream-based-photography-recomendation-main/tests/test_web.py)
- [tests/test_workflow.py](D:/Project/video-stream-based-photography-recomendation-main/tests/test_workflow.py)
- [video_guide/cli.py](D:/Project/video-stream-based-photography-recomendation-main/video_guide/cli.py)
- [video_guide/core/__init__.py](D:/Project/video-stream-based-photography-recomendation-main/video_guide/core/__init__.py)
- [video_guide/core/backends.py](D:/Project/video-stream-based-photography-recomendation-main/video_guide/core/backends.py)
- [video_guide/core/graph.py](D:/Project/video-stream-based-photography-recomendation-main/video_guide/core/graph.py)
- [video_guide/core/models.py](D:/Project/video-stream-based-photography-recomendation-main/video_guide/core/models.py)
- [video_guide/core/service.py](D:/Project/video-stream-based-photography-recomendation-main/video_guide/core/service.py)
- [video_guide/core/state.py](D:/Project/video-stream-based-photography-recomendation-main/video_guide/core/state.py)
- [video_guide/core/vision.py](D:/Project/video-stream-based-photography-recomendation-main/video_guide/core/vision.py)
- [video_guide/static/index.html](D:/Project/video-stream-based-photography-recomendation-main/video_guide/static/index.html)
- [video_guide/static/studio.js](D:/Project/video-stream-based-photography-recomendation-main/video_guide/static/studio.js)
- [video_guide/web.py](D:/Project/video-stream-based-photography-recomendation-main/video_guide/web.py)

## 3. Added Files

PhotoAgent：

- [photography_viewpoint_agent/schemas/handoff.py](D:/Project/PhotoAgent/photography_viewpoint_agent/schemas/handoff.py)
- [photography_viewpoint_agent/integration/__init__.py](D:/Project/PhotoAgent/photography_viewpoint_agent/integration/__init__.py)
- [photography_viewpoint_agent/integration/guidance_adapter.py](D:/Project/PhotoAgent/photography_viewpoint_agent/integration/guidance_adapter.py)
- [photography_viewpoint_agent/graph/integrated_workflow.py](D:/Project/PhotoAgent/photography_viewpoint_agent/graph/integrated_workflow.py)
- [tests/test_integration.py](D:/Project/PhotoAgent/tests/test_integration.py)
- [tools/run_integrated_demo.py](D:/Project/PhotoAgent/tools/run_integrated_demo.py)
- [docs/INTEGRATION_DELIVERY.md](D:/Project/PhotoAgent/docs/INTEGRATION_DELIVERY.md)

video_guide：

- [tests/conftest.py](D:/Project/video-stream-based-photography-recomendation-main/tests/conftest.py)

## 4. Removed / Deprecated

删除下游下列文件（可从原始备份查看）：

- [video_guide/preprocessing/__init__.py](D:/Project/integration-backup-20260917/video_guide/preprocessing/__init__.py)
- [video_guide/preprocessing/clip/__init__.py](D:/Project/integration-backup-20260917/video_guide/preprocessing/clip/__init__.py)
- [video_guide/preprocessing/composition/__init__.py](D:/Project/integration-backup-20260917/video_guide/preprocessing/composition/__init__.py)
- [video_guide/preprocessing/pipeline.py](D:/Project/integration-backup-20260917/video_guide/preprocessing/pipeline.py)
- [video_guide/preprocessing/target/__init__.py](D:/Project/integration-backup-20260917/video_guide/preprocessing/target/__init__.py)

同时移除 CropBox、locate_crop、render_crop、crop-based zoom estimation、旧 demo CLI 与 Web 自动随机目标入口。
原 crop 专用测试已替换为新 contract / phase / session 测试；没有通过跳过旧测试保留双目标流程。

**crop-based composition generation has been replaced by PhotoAgent Target Sketch.**

下游原目录没有 .git；修改前源码、tests、docs、README 和 pyproject 已备份到 `D:/Project/integration-backup-20260917`。

## 5. Handoff Contract

```python
TargetPackage(
    video_path: str,
    reference_frame: FrameInfo,
    current_frame: FrameInfo,
    target_state: TargetState,
    target_sketch_path: str,
    render_meta: RenderMeta,
)
```

上游使用既有 Pydantic Schema；build_target_package 拒绝上游 error 与 failed validation，仅导出稳定字段。
Adapter 转为下游 GuideGraphInput：

```python
{
    "video_path": package.video_path,
    "current_frame": package.current_frame.path,  # 后续 step 可替换
    "reference_frame": package.reference_frame.path,
    "target_sketch": package.target_sketch_path,
    "target_state": package.target_state.model_dump(mode="json"),
    "render_meta": package.render_meta.model_dump(mode="json"),
    "session_directory": "...",
    "session_id": "...",  # 可选
    "step_id": 1,         # 可选，省略自动分配
    "video_url": None,    # 可选
}
```

GuideResult：phase、reference_reached、target_reached、actions、guidance、confidence、warnings、evidence。
Action：actor(camera/subject/lens)、action、可选 magnitude(small/medium/large)、reason。禁止 actor/action 错配和阶段标志矛盾。

Session target 只归档一次，图像内容与元数据锁定；每个 guidance step 只保存 current frame、result、manifest 和执行标识，不重复复制视频/目标。
上游调试产物位于 session/generation/，锁定目标位于 session/target/，逐步结果位于 session/guidance/step_XXXXXX/。
视频使用稳定原路径；Web 上传视频保存在 session/source/ 一次。目标变化必须新建 session。

## 6. Final Graph

```text
START
  |
  v
 generate_target  [既有 PhotoAgent Graph]
  |
  v
 TargetPackage -> build_handoff
  |
  v
 guide_once      [独立 Guidance Graph]
  |                  |
  |           validate_input
  |                  |
  |           prepare_inputs
  |                  |
  |           analyze_alignment
  |           navigation / composition / reached / uncertain
  |                  |
  |           persist_result
  v
 END -> reference_frame / current_frame / target_state /
        target_sketch / render_meta / guidance / target_package
```

后续帧仅复用 TargetPackage 调用 Guidance；不重跑 Target Generation。父图和子图均支持检查点恢复；每个新 guidance step 使用新 checkpoint thread_id。

## 7. Tests

2026-09-17，PhotoAgent .venv311 / Python 3.11.16：

| 检查 | 结果 |
|---|---|
| PhotoAgent `python -m pytest -q -rs` | 186 passed，11 skipped |
| video_guide `python -m pytest -q` | 42 passed |
| Python compileall | 通过 |
| 上下游 contract/graph/Web import | 通过 |
| studio.js `node --check` | 通过 |
| pip check | No broken requirements found |
| 离线真实视频 → Target Generation → Guidance demo | reached，两个 reached=true，actions=[] |

11 个 skip 全部来自原有 test_mesh_gpu.py，环境缺少可用的 CUDA/PyTorch3D 构建。
下游有一个第三方 Starlette/AnyIO DeprecationWarning，不影响测试结果。

新测试覆盖：handoff/JSON 往返、framing、subject reposition、扩面 padding、rotation/depth_3d/depth_mesh 元数据、四种 phase、非法动作/结果拒绝、VLM 图像/视频/JSON 传输、瞬时重试、永久错误不重试、父图和子图检查点恢复、锁定目标复用/篡改拒绝、重复编号、并发 step 分配、CLI/Web 新输入与文件路径保护。

首轮端到端测试的高频随机图因有损视频/JPEG 编码使像素误差超过本地匹配阈值；测试改用稳定的渐变小视频后通过，未放宽匹配阈值。

可运行 demo：

```powershell
cd D:\Project\PhotoAgent
.\.venv311\Scripts\python tools/run_integrated_demo.py --output workdir/integrated-demo-new
```

本次产物：

- [workdir/integrated-demo-20260917/integrated_result.json](D:/Project/PhotoAgent/workdir/integrated-demo-20260917/integrated_result.json)
- [workdir/integrated-demo-20260917/session/guidance/step_000001/result.json](D:/Project/PhotoAgent/workdir/integrated-demo-20260917/session/guidance/step_000001/result.json)


## 8. Remaining Issues

- 真实在线 VLM 未调用；prompt/schema/transport 通过离线 mock 验证，空间动作的实际准确率尚未评估。
- LocalBackend 只作保守离线验证；人物重排、padding 和深度构图不会伪造确定动作，会返回 uncertain。
- 本环境未验证 CUDA/PyTorch3D 渲染；此部分上游实现未修改。
- Web 每次提交创建新 session 第一阶段；连续 step 已通过 Python/CLI 支持，摄像头循环和 target refresh 按本次范围未实现。
- 引用的原视频必须保持可读且内容不变；没有为了归档而重复复制完整视频。
