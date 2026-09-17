# Qwen3.8 视频输入配置指南

本工程默认模型名为 `qwen3.8-max`。官方已列出其视频支持：[视觉模型](https://help.aliyun.com/zh/model-studio/vision-model)。你也可以填写当前地域实际可调用的 Qwen3.8 模型 ID。

## 1. 获取三个接入信息

在阿里云百炼控制台选择地域和业务空间，获取 API Key、兼容接口地址以及可访问的模型 ID。Key、域名和模型权限需要对应同一地域/空间，具体入口见 [获取 API Key](https://help.aliyun.com/zh/model-studio/get-api-key)。

这里的“兼容服务”是百炼提供的 HTTP API，不需要你自己搭建一个模型服务。项目里的 Python 后端负责向它发送请求；浏览器只向本机 Python 服务上传文件。

## 2. 编辑配置文件

打开工程根目录的 `config/vlm.json`：

```json
{
  "endpoint": "https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions",
  "model": "qwen3.8-max",
  "api_key": "填写你的实际 API Key",
  "fps": 2.0,
  "timeout": 120
}
```

上面的 WorkspaceId 是占位符，必须替换，其他地域请使用其对应域名。如果控制台提供北京公共兼容 Base URL `https://dashscope.aliyuncs.com/compatible-mode/v1`，则完整 endpoint 填 `https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions`。以你控制台的接入信息为准。

**为什么必须填完整地址？** 官方 SDK 的 `base_url` 只填到 `/v1`，SDK 会自动追加 `/chat/completions`。本项目使用 Python HTTP 请求直接 POST 到 endpoint，因此不会自动追加路径。不要填写控制台网页地址，也不要填 DashScope 原生 `/services/aigc/...` 接口；它的消息结构不同。

| 字段 | 用途 |
|---|---|
| endpoint | 完整 POST 地址，必须以 `/chat/completions` 结尾 |
| model | 服务端识别的模型 ID，例如 `qwen3.8-max` |
| api_key | 请求 `Authorization: Bearer ...` 使用的密钥 |
| fps | 服务端视频采样频率，默认 2，即通常每 0.5 秒一帧 |
| timeout | 单次 HTTP 超时，单位秒，默认 120 |

JSON 不支持注释或末尾多余逗号。配置文件保存为 UTF-8。仅 `config/vlm.json` 用于本地密钥，示例文件不要放真实密钥。配置不会通过 Web 文件接口暴露；结果文件不保存密钥或有签名的完整视频 URL。

## 3. 保存后如何生效

- **改配置文件**：Web 的下一次 VLM 分析自动重新读取，无需重启；正在运行的任务不会改变。CLI 下一次执行即读取。
- **从不同目录启动**：默认相对当前工作目录查找 `config/vlm.json`。建议在工程根目录运行；CLI 可用 `--config D:/projects/video-streaming-recommendation/config/vlm.json`，Web 可通过 `VLM_CONFIG` 指定绝对路径。
- **直接集成 Python**：每次创建 `VLMBackend()` 时读取配置；如果复用旧实例，需要重新创建实例以应用文件修改。
- **修改 Python 代码**：需要重启服务，与配置文件热读取是两回事。

本地启动：

```powershell
cd D:\projects\video-streaming-recommendation
python -m uvicorn video_guide.web:app --host 127.0.0.1 --port 8000
```

浏览器选择 Qwen VLM，上传视频、Current Frame、Reference Frame、Target Sketch，填写 TargetState/RenderMeta JSON，再点击分析。仅填写配置不会产生模型请求，点击分析才调用。

## 4. 仍然可以使用环境变量

优先级为：Python 构造参数 > `VLM_*` 环境变量 > JSON 配置 > 内置默认值。API Key 为空时额外尝试 `DASHSCOPE_API_KEY`。

字段映射：`VLM_ENDPOINT`、`VLM_MODEL`、`VLM_API_KEY`、`VLM_FPS`、`VLM_TIMEOUT`。

```powershell
$env:VLM_API_KEY = "你的实际 API Key"
$env:VLM_FPS = "2"
python -m uvicorn video_guide.web:app --host 127.0.0.1 --port 8000
```

这只设置当前 PowerShell 及其随后启动的子进程。另一终端里已经启动的 Web 服务不会收到变化；需要停止旧服务，再从设置过变量的终端启动。修改 Windows 用户/系统环境变量也不会更新已运行进程，通常需重新打开终端再启动服务。

如果曾设置过 `VLM_FPS` 等变量，它们会覆盖配置文件，导致看起来“改文件没生效”。请从启动服务的终端移除相应覆盖后重启，例如：

```powershell
Remove-Item Env:VLM_FPS -ErrorAction SilentlyContinue
```

工程不会自动加载 `.env`；`config/vlm.json` 才是直接可编辑并自动读取的配置入口。

## 5. 实际如何传入视频

请求用户消息包含文字、三张参考图和一个完整视频。视频部分按[官方图像与视频文档](https://help.aliyun.com/zh/model-studio/vision)构造：

```json
{
  "type": "video_url",
  "video_url": {"url": "data:video/mp4;base64,完整视频的编码内容"},
  "fps": 2.0
}
```

`fps` 与 `video_url` 同级。源文件的每个字节都编码上传，不经过客户端抽帧、重新编码或联系表转换。服务端仍会按 fps 采样处理，因此“直接输入视频”不等于模型逐帧分析原始 30 FPS。图片使用 `image_url`；仅支持 image_url 的服务不足以运行本后端。

Web 主模块发送用户上传的原视频；目标图由 PhotoAgent 提供。独立调用发送 GuideInput.video_path 指定的视频。三张图与完整 TargetState/RenderMeta 一起输入，不执行 crop 目标生成。提示词明确以当前帧参考图为准，不将原视频末帧当作当前帧。

### 大视频

Base64 编码后上限为 100 MB（原始文件约 75 MB），本工程在发送前检查，不会偷偷抽帧规避。较大视频使用公网/OSS URL，填在 Web 的“原视频 URL”或 CLI 的 `--video-url`。仍需本地视频作为稳定输入引用，URL 必须对应同一视频，程序无法自动验证远端内容一致性。

模型服务须能访问该 URL，不能使用 `localhost`、本地文件路径或 OSS 内网域名。URL 应返回正确的 Content-Type 和 Content-Length。工程不会自动上传 OSS；临时签名 URL 也由你提供，不会写入结果归档。

已下载的 TUM 样本中，`rpy`（7.07 MB）、`xyz`（7.69 MB）和 `room`（13.13 MB）均可直接 Base64 编码传输。它们的原始官方 URL 记录在 `data/tum-mini/manifest.json`，可以复制到视频 URL 字段（第三方下载服务是否可访问仍由实际请求验证）。

## 常见错误

- endpoint 占位符：替换 WorkspaceId，或使用控制台实际完整地址。
- 401/403：检查 Key、地域、业务空间及模型访问权限。
- 404/模型不存在：检查完整 endpoint 和当前地域的准确模型 ID。
- 100 MB：改用与原视频相同的公网 URL。
- 视频下载失败：检查 URL 是否公网可访问及是否过期。
- JSON 结构不符合要求：模型输出校验失败，任务不会发布无效 guidance step。

本次验证覆盖配置读取、优先级、原视频字节传输、FPS 参数、Web 归档；尚未使用真实 API Key 调用云模型。
