# Guidance LangGraph integration

`build_guide_graph(backend=None, checkpointer=None)` 返回独立编译子图，输入 `GuideGraphInput`，输出 `GuideGraphOutput(result, output_directory)`。

```text
START -> validate_input -> prepare_inputs -> analyze_alignment -> persist_result -> END
```

父图可以用 adapter 在节点中调用子图，或在输入字段相同的情况下直接 `add_node("guide", build_guide_graph(...))`。
PhotoAgent 的 `graph/integrated_workflow.py` 使用显式 TargetPackage，内部 AgentState 不跨边界。

## Checkpoint / retry

```python
from langgraph.checkpoint.memory import InMemorySaver
from video_guide.core import build_guide_graph

graph = build_guide_graph(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "session-1-step-1"}}
# inputs 使用 GUIDANCE.md 中的完整 GuideGraphInput。
result = graph.invoke(inputs, config)
# 节点失败修复后：
# result = graph.invoke(None, config)
```

每个新 guidance step 使用新 thread_id；恢复已有 step 使用原 thread_id 和 invoke(None)。不要在已完成 thread 上省略字段重新开始。
持久化失败可从 persist_result 恢复，不重复模型调用。发布节点重放通过 execution_id 校验，只认可同一执行的结果。
模型只对 transient_model_error 认可的瞬时故障重试，最多两次；不合法的结果/输入直接失败。
API Key、客户端实例和 Base64 不进入 graph state。`video_url` 可含签名，会存在调用输入和 checkpointer 中，但不写进磁盘 manifest；生产部署应妥善保护 checkpoint 存储。

## Session target

首次输入以图片内容 hash 和完整元数据锁定目标；目标通过独立临时目录原子发布。后续步骤拒绝替换目标或篡改归档。
视频仅引用源路径，不按 step 复制；调用方负责保持视频文件不变、可读。session_id 只是会话标识，session_directory 是明确的会话根目录。
目标变更需要新的 session；本版本不实现 target refresh 或摄像头循环。

Breaking change: crop-based composition generation has been replaced by PhotoAgent Target Sketch. 旧 crop 分支、run/ 单次归档模型和旧输入/结果字段已移除，迁移方式见 GUIDANCE.md。
