# 滑雪视频真实 VLM 通路测试

> 历史测试/交付记录。当前两个仓库均包含完整上下游源码，安装方式以根目录 README 为准；以下本地路径指向当时的测试产物。

- 日期：2026-09-17。
- 模型：glm-5.3-flash。
- 输入：原滑雪视频，上游使用 examples/11_frame_only__zoom_out_center.json（扩面取景）。
- 通路：真实视频 → PhotoAgent Graph → TargetPackage → Guidance Graph → 真实 HTTP 模型请求 → 严格 schema 解析 → 原子落盘。
- 成功请求：三张图（Current/Reference/Target Sketch）+ 全部 TargetState/RenderMeta，input_mode=images，reasoning_effort=low。
- 输出：navigation，reference_reached=false，target_reached=false；camera.move_forward、camera.pan_right。
- 总耗时：11.54 秒；成功模型请求：9.84 秒；HTTP 200。
- 仅验证通路，不评价动作准确性。

## 适配过程

完整视频 Base64 请求约 44.8 MB，两次连接重置（尚不能确定是网关大小限制还是其他传输问题）。小请求成功，确认 endpoint 和 Key 有效。
改为显式 images 请求后，服务返回 400：模型始终思考，不支持关闭思考。使用 reasoning_effort=low 并省略 enable_thinking 后成功。
完整视频直接传入此服务的能力仍未验证通过。默认 video 后端保留，无自动降级。

## 验证

- 下游完整 pytest：43 passed。
- 上游集成测试：11 passed。
- 新增 images 模式测试，确保三图/结构元数据保留、视频不读取、不发送 enable_thinking=false。
- API Key 仅用于进程内请求，未保存到文件。

## 产物

- [integrated_result.json](D:/Project/PhotoAgent/workdir/skiing-glm53-live-pass-20260917/integrated_result.json)
- [test_summary.json](D:/Project/PhotoAgent/workdir/skiing-glm53-live-pass-20260917/test_summary.json)
- [requests.json](D:/Project/PhotoAgent/workdir/skiing-glm53-live-pass-20260917/requests.json)
- [session/guidance/step_000001/result.json](D:/Project/PhotoAgent/workdir/skiing-glm53-live-pass-20260917/session/guidance/step_000001/result.json)
