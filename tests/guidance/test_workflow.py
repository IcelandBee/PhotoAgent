import json
from dataclasses import fields
from pathlib import Path
import pytest
from PIL import Image
from video_guide.core import GuideInput, GuideService
from video_guide.core.backends import LocalBackend, VLMBackend
from .conftest import TARGET, META


def test_service_and_local_reached(payload):
    inputs = GuideInput(**{f.name: payload[f.name] for f in fields(GuideInput) if f.name in payload})
    result, folder = GuideService().run(inputs, Path(payload["session_directory"]))
    assert result.phase == "reached"
    assert result.actions == []
    assert (folder / "result.json").is_file()


def test_local_uncertainty():
    black, white = Image.new("RGB", (100, 80)), Image.new("RGB", (100, 80), "white")
    result = LocalBackend().analyze(black, white, white, None, TARGET, META)
    assert result.phase == "uncertain" and not result.reference_reached
    assert result.actions == []
    result = LocalBackend().analyze(white, white, black, None, TARGET, META)
    assert result.phase == "uncertain" and result.reference_reached

@pytest.mark.parametrize("change", [
    {"reference_reached": "true"}, {"confidence": True}, {"confidence": float("nan")},
    {"target_reached": True}, {"actions": []}, {"guidance": ""}, {"warnings": "bad"},
    {"crop_box": {}}, {"phase": "unknown"},
    {"actions": [{"actor": "subject", "action": "zoom_in"}]},
    {"actions": [{"actor": "lens", "action": "zoom_in", "magnitude": "huge"}]},
])
def test_vlm_schema_rejects_invalid(change):
    valid = {"phase": "composition", "reference_reached": True, "target_reached": False,
             "actions": [{"actor": "lens", "action": "zoom_in"}], "guidance": "适当变焦", "confidence": .9}
    assert VLMBackend.parse(json.dumps(valid)).phase == "composition"
    with pytest.raises(ValueError):
        VLMBackend.parse(json.dumps({**valid, **change}))


def test_navigation_cannot_move_subject():
    with pytest.raises(ValueError):
        VLMBackend.parse(json.dumps({"phase": "navigation", "reference_reached": False, "target_reached": False,
          "actions": [{"actor": "subject", "action": "subject_left"}], "guidance": "移动", "confidence": .8}))

@pytest.mark.parametrize("change", [{"session_id": "../bad"}, {"step_id": True}, {"step_id": 0}, {"target_state": {}}, {"render_meta": {}}, {"current_frame": "missing"}])
def test_invalid_input(payload, change):
    args = {f.name: payload[f.name] for f in fields(GuideInput) if f.name in payload}
    with pytest.raises(ValueError):
        GuideInput(**{**args, **change}).validate()


def test_cli_uses_target_contract(payload, tmp_path, capsys):
    from video_guide.cli import main
    target_file, meta_file = tmp_path / "target.json", tmp_path / "meta.json"
    target_file.write_text(json.dumps(payload["target_state"]), encoding="utf-8")
    meta_file.write_text(json.dumps(payload["render_meta"]), encoding="utf-8")
    args = ["guide"]
    for field in ("video_path", "current_frame", "reference_frame", "target_sketch", "session_directory"):
        args.extend(["--" + field.replace("_", "-"), payload[field]])
    args.extend(["--target-state", str(target_file), "--render-meta", str(meta_file)])
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["phase"] == "reached"
