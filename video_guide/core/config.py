"""集中配置；每次创建后端读取，环境变量覆盖文件。"""
import json
import os
import math
from pathlib import Path
from urllib.parse import urlsplit


def load_config(path=None):
    explicit = path or os.getenv("VLM_CONFIG")
    source = Path(explicit or "config/vlm.json")
    values = {"endpoint": "", "model": "qwen3.8-max", "api_key": "", "fps": 2.0, "timeout": 120.0}
    if source.is_file():
        try:
            data = json.loads(source.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as error:
            raise ValueError("VLM 配置文件无法读取或不是有效 JSON") from error
        if not isinstance(data, dict) or set(data) - set(values):
            raise ValueError("VLM 配置必须为对象且只能包含 endpoint/model/api_key/fps/timeout")
        values.update(data)
    elif explicit:
        raise ValueError("指定的 VLM 配置文件不存在")
    for key in values:
        name = "VLM_" + key.upper()
        if name in os.environ:
            values[key] = os.environ[name]
    if not values["api_key"]:
        values["api_key"] = os.getenv("DASHSCOPE_API_KEY", "")
    return values


def validate_config(values):
    endpoint = values["endpoint"]
    if not isinstance(endpoint, str) or "{" in endpoint or "}" in endpoint:
        raise ValueError("请在 config/vlm.json 填写实际 endpoint，替换 WorkspaceId 占位符")
    parsed = urlsplit(endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or not parsed.path.endswith("/chat/completions") or parsed.query or parsed.fragment or parsed.username:
        raise ValueError("endpoint 必须是完整的 http(s) 请求地址，以 /chat/completions 结尾")
    if not isinstance(values["model"], str) or not values["model"].strip():
        raise ValueError("model 不能为空")
    if not isinstance(values["api_key"], str):
        raise ValueError("api_key 必须为字符串")
    for key, lower, upper in (("fps", .1, 10), ("timeout", 1, 3600)):
        raw = values[key]
        try:
            number = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{key} 必须是数值") from None
        if isinstance(raw, bool) or not math.isfinite(number) or not lower <= number <= upper:
            raise ValueError(f"{key} 必须在 {lower} 到 {upper} 之间")
        values[key] = number
    return values
