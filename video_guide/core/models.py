"""Guidance wire contract. No dependency on the generator's internal state."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal
import json
import math
import re


def validate_viewpoint_intent(viewpoint):
    """Shared guard for graph input and direct backend calls."""
    if not isinstance(viewpoint, dict):
        raise ValueError("viewpoint must be an orientation object")
    if set(viewpoint) - {"mode", "yaw_deg", "pitch_deg", "roll_deg"}:
        raise ValueError("viewpoint accepts orientation only: mode/yaw_deg/pitch_deg/roll_deg; no translation or renderer settings")
    if viewpoint.get("mode", "none") not in ("none", "rotation"):
        raise ValueError("viewpoint.mode must be none or rotation")
    angles = [viewpoint.get(axis, 0) for axis in ("yaw_deg", "pitch_deg", "roll_deg")]
    if any(type(v) not in (int, float) or not math.isfinite(v) or not -15 <= v <= 15 for v in angles):
        raise ValueError("Viewpoint angles must be finite degrees in [-15, 15]")
    if viewpoint.get("mode", "none") == "none" and any(angles):
        raise ValueError("Nonzero angles require viewpoint.mode=rotation")


@dataclass(frozen=True)
class GuideInput:
    video_path: str | Path
    current_frame: str | Path
    reference_frame: str | Path
    target_sketch: str | Path
    target_state: dict
    render_meta: dict
    session_id: str | None = None
    step_id: int | None = None
    video_url: str | None = None

    def validate(self):
        for name in ("video_path", "current_frame", "reference_frame", "target_sketch"):
            if not Path(getattr(self, name)).is_file():
                raise ValueError(f"输入文件不存在：{name}")
        for name in ("target_state", "render_meta"):
            value = getattr(self, name)
            if not isinstance(value, dict) or not value:
                raise ValueError(f"{name} must be a nonempty JSON object")
            json.dumps(value, allow_nan=False)
        subject = self.target_state.get("subject", {})
        framing = self.target_state.get("framing", {})
        viewpoint = self.target_state.get("viewpoint", {})
        if not all(isinstance(value, dict) for value in (subject, framing, viewpoint)):
            raise ValueError("subject/framing/viewpoint must be JSON objects")
        if subject.get("mode") not in ("follow_reference", "reposition"):
            raise ValueError("target_state.subject.mode is required")
        def box(value, normalized=False):
            if not isinstance(value, (list, tuple)) or len(value) != 4 or any(
                type(v) not in (int, float) or not math.isfinite(v) for v in value
            ) or value[0] >= value[2] or value[1] >= value[3]:
                raise ValueError("Invalid viewport/bbox")
            if normalized and not all(0 <= v <= 1 for v in value):
                raise ValueError("Subject bbox must be normalized")
        box(framing.get("reference_viewport"))
        box(self.render_meta.get("rendered_viewport"))
        padding = self.render_meta.get("padding", [0, 0, 0, 0])
        if not isinstance(padding, (list, tuple)) or len(padding) != 4 or any(type(v) is not int or v < 0 for v in padding):
            raise ValueError("padding must contain four nonnegative pixel counts")
        if subject["mode"] == "reposition":
            box(subject.get("bbox"), True)
        elif subject.get("bbox") is not None:
            raise ValueError("follow_reference requires bbox=null")
        validate_viewpoint_intent(viewpoint)
        if self.session_id is not None and (not isinstance(self.session_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", self.session_id)):
            raise ValueError("Invalid session_id")
        if self.step_id is not None and (type(self.step_id) is not int or self.step_id < 1):
            raise ValueError("step_id must be a positive integer")


_ACTIONS = {
    "camera": {"move_left", "move_right", "move_forward", "move_backward", "move_up", "move_down", "pan_left", "pan_right", "tilt_up", "tilt_down", "roll_left", "roll_right"},
    "subject": {"subject_left", "subject_right", "subject_forward", "subject_backward"},
    "lens": {"zoom_in", "zoom_out"},
}

@dataclass(frozen=True)
class GuidanceAction:
    actor: Literal["camera", "subject", "lens"]
    action: str
    magnitude: Literal["small", "medium", "large"] | None = None
    reason: str | None = None

    def __post_init__(self):
        if self.actor not in _ACTIONS or self.action not in _ACTIONS[self.actor]:
            raise ValueError("Invalid actor/action pair")
        if self.magnitude not in (None, "small", "medium", "large"):
            raise ValueError("Invalid action magnitude")
        if self.reason is not None and not isinstance(self.reason, str):
            raise ValueError("reason must be text")


@dataclass
class GuideResult:
    phase: Literal["navigation", "composition", "reached", "uncertain"]
    reference_reached: bool
    target_reached: bool
    actions: list[GuidanceAction]
    guidance: str
    confidence: float
    warnings: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)

    def __post_init__(self):
        if type(self.reference_reached) is not bool or type(self.target_reached) is not bool:
            raise ValueError("reached flags must be booleans")
        flags = {"navigation": (False, False), "composition": (True, False), "reached": (True, True)}
        if self.phase not in (*flags, "uncertain"):
            raise ValueError("Invalid phase")
        if self.phase in flags and (self.reference_reached, self.target_reached) != flags[self.phase]:
            raise ValueError("phase and reached flags disagree")
        if self.phase == "uncertain" and self.target_reached:
            raise ValueError("uncertain cannot mean target reached")
        if not isinstance(self.actions, list) or not all(isinstance(a, GuidanceAction) for a in self.actions):
            raise ValueError("actions must contain GuidanceAction objects")
        if self.phase in ("reached", "uncertain") and self.actions:
            raise ValueError("reached/uncertain must have no actions")
        if self.phase in ("navigation", "composition") and not self.actions:
            raise ValueError("Actionable phase requires actions; otherwise use uncertain")
        if self.phase == "navigation" and any(a.actor != "camera" for a in self.actions):
            raise ValueError("Navigation must adjust the camera first")
        if not isinstance(self.guidance, str) or not self.guidance.strip():
            raise ValueError("guidance must be nonempty text")
        if type(self.confidence) not in (int, float) or not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        if not isinstance(self.warnings, list) or not all(isinstance(w, str) for w in self.warnings):
            raise ValueError("warnings must be a list of strings")
        if not isinstance(self.evidence, dict):
            raise ValueError("evidence must be an object")
        json.dumps(self.to_dict(), allow_nan=False)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        try:
            data = dict(data)
            data["actions"] = [GuidanceAction(**a) for a in data["actions"]]
            return cls(**data)
        except (KeyError, TypeError) as error:
            raise ValueError("Invalid guidance result schema") from error
