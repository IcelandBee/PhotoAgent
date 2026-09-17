import base64
import io
import json
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit
from .config import load_config, validate_config
from .models import GuidanceAction, GuideResult, _ACTIONS
from .vision import compare, alignment, corners


class LocalBackend:
    """Conservative offline image evidence; never infer 3D motion from a 2D crop."""
    def analyze(self, current, reference, target_sketch, video, target_state, render_meta, video_url=None):
        reached, confidence, evidence = compare(current, reference)
        warnings = ["本地匹配无法可靠恢复三维位移、人物动作或真实焦距；复杂构图请使用 VLM。"]
        if not reached:
            matrix, _ = alignment(reference, current)
            actions = []
            if matrix is not None:
                dx, dy = corners(reference, matrix).mean(axis=0) / current.size - 0.5
                if abs(dx) > .04:
                    actions.append(GuidanceAction("camera", "pan_right" if dx > 0 else "pan_left", "small", "参考视角在当前图像中的位置"))
                if abs(dy) > .04:
                    actions.append(GuidanceAction("camera", "tilt_down" if dy > 0 else "tilt_up", "small"))
            return GuideResult("navigation" if actions else "uncertain", False, False, actions,
                               "小幅调整相机朝向后重新比较参考画面。" if actions else "无法可靠确定方向，请核对参考场景。",
                               confidence, warnings, evidence)
        # Homography can match the background despite a misplaced subject. Only
        # close full-image appearance supports the offline reached decision.
        _, target_confidence, target_evidence = compare(current, target_sketch)
        target_reached = target_evidence.get("mean_pixel_error", 1) < .015
        # Synthetic padding/warps cannot establish a physically reached target.
        synthetic = (any(render_meta.get("padding", [])) or
                     target_state.get("viewpoint", {}).get("mode", "none") != "none" or
                     target_state["subject"]["mode"] == "reposition")
        target_reached = target_reached and not synthetic
        return GuideResult("reached" if target_reached else "uncertain", True, target_reached, [],
                           "已接近目标画面，可以拍摄。" if target_reached else "已接近参考视角；目标构图需要进一步语义判断。",
                           min(confidence, target_confidence), warnings,
                           {"reference": evidence, "target": target_evidence})


class VLMBackend:
    """OpenAI-compatible VLM with explicit video or images-only transport."""

    def __init__(self, endpoint=None, model=None, api_key=None, timeout=None, fps=None, config_path=None, input_mode="video", reasoning_effort=None):
        if input_mode not in ("video", "images"):
            raise ValueError("input_mode must be video or images")
        if reasoning_effort not in (None, "low", "high", "max"):
            raise ValueError("Unsupported reasoning_effort")
        self.reasoning_effort = reasoning_effort
        self.input_mode = input_mode
        values = load_config(config_path)
        for key, value in dict(endpoint=endpoint, model=model, api_key=api_key, timeout=timeout, fps=fps).items():
            if value is not None:
                values[key] = value
        for key, value in validate_config(values).items():
            setattr(self, key, value)

    def analyze(self, current, reference, target_sketch, video, target_state, render_meta, video_url=None):
        instruction = """你是实拍取景指导助手。三图依次为 Current Frame、Reference Frame、Target Sketch。
Reference 是优秀历史真实视角；Target Sketch 是最终理想构图，可能扩面、人物重排或深度视点变换，绝不假设它是任何图中的裁剪区域。
先判断 Current 是否接近 Reference 视角。未接近时 phase=navigation，只输出摄影师相机移动/朝向动作，禁止提前指导人物或变焦。
接近后比较 Current 与 Target Sketch，结合 TargetState 和 RenderMeta 输出 composition 动作；达到目标时 reached 且 actions=[]。
reference_reached 表示已进入参考视角附近、可以执行目标调整；不要把目标要求的后退/平移再次误判为需要返回原机位。
subject.mode=reposition 明确表示人物目标站位/尺度改变，应结合 source/natural/rendered/target bbox 判断人物动作，不能转换成裁图。
reference_viewport 超出 [0,1] 或 padding>0 是合法扩面意图，可通过后退、zoom_out 或取景调整实现；填充像素不是需要复现的真实物体。
viewpoint 的 translation_x/y/z 表示三维相机位置变化（原参考相机坐标系），不是二维裁图；结合现场证据判断动作方向，不凭符号盲猜。
render_meta 描述实际渲染后的视窗、padding、subject bbox 和 viewpoint_warp。以当前图为准，不把视频末帧当作当前帧。
方向左右均以摄影师当前画面为准。无法确定可执行动作时 uncertain，target_reached=false，actions=[]；不编造距离或焦距。
只返回 JSON: {"phase":"navigation|composition|reached|uncertain","reference_reached":bool,"target_reached":bool,
"actions":[{"actor":"camera|subject|lens","action":"允许的动作","magnitude":"small|medium|large或null","reason":"依据或null"}],
"guidance":"中文实拍指引","confidence":0到1,"warnings":["不确定性"]}。
navigation 两个 reached=false；composition reference_reached=true,target_reached=false；reached 两者=true。
"""
        instruction += "\n允许的 actor/actions: " + json.dumps({k: sorted(v) for k, v in _ACTIONS.items()})
        instruction += "\nTargetState: " + json.dumps(target_state, ensure_ascii=False, allow_nan=False)
        instruction += "\nRenderMeta: " + json.dumps(render_meta, ensure_ascii=False, allow_nan=False)
        instruction += "\n本次输入包含完整视频。" if self.input_mode == "video" else "\n本次仅提供三张图和结构数据，不提供视频；请仅依据这些输入判断。"
        content = [{"type": "text", "text": instruction}]
        for label, image in zip(("Current Frame", "Reference Frame", "Target Sketch"), (current, reference, target_sketch)):
            image = image.copy()
            image.thumbnail((1536, 1536))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            content.extend([{"type": "text", "text": label}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()}}])
        if self.input_mode == "video":
            if video_url:
                parsed = urlsplit(video_url)
                if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username:
                    raise ValueError("视频 URL 必须是模型服务可访问的 http(s) 地址")
                url = video_url
            else:
                video = Path(video)
                if 4 * ((video.stat().st_size + 2) // 3) >= 100_000_000:
                    raise ValueError("视频 Base64 编码将超过 100 MB 上限；请填写该原视频的公网/OSS URL（CLI: --video-url），不会自动截帧或压缩视频")
                mime = {".mp4": "video/mp4", ".avi": "video/x-msvideo", ".mov": "video/quicktime", ".mkv": "video/x-matroska", ".webm": "video/webm"}.get(video.suffix.lower())
                if mime is None:
                    raise ValueError("不支持的视频文件扩展名")
                url = f"data:{mime};base64," + base64.b64encode(video.read_bytes()).decode()
            content.extend([{"type": "text", "text": "完整输入视频"}, {"type": "video_url", "video_url": {"url": url}, "fps": self.fps}])
        payload = {"model": self.model, "messages": [{"role": "user", "content": content}], "temperature": 0, "enable_thinking": False, "response_format": {"type": "json_object"}}
        if self.reasoning_effort is not None:
            payload.pop("enable_thinking")
            payload["reasoning_effort"] = self.reasoning_effort
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode(), headers=headers, method="POST")
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            envelope = json.load(response)
        request_seconds = round(time.perf_counter() - started, 2)
        raw = envelope["choices"][0]["message"]["content"].strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = self.parse(raw)
        result.evidence.update(model=self.model, fps=self.fps, video_transport=("url" if video_url else "base64") if self.input_mode == "video" else "none", input_mode=self.input_mode, vlm_request_seconds=request_seconds)
        return result

    @staticmethod
    def parse(raw):
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("VLM result must be a JSON object")
        result = GuideResult.from_dict(value)
        result.evidence["method"] = "vlm"
        return result
