import io
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from video_guide import web
from video_guide.core.backends import VLMBackend
from .conftest import TARGET, META


def test_web_new_contract_and_artifacts(tmp_path):
    buffer = io.BytesIO()
    Image.new("RGB", (100, 80), "green").save(buffer, format="PNG")
    png = buffer.getvalue()
    files = {name: (name + ".png", png, "image/png") for name in ("current_frame", "reference_frame", "target_sketch")}
    files["video"] = ("sample.avi", b"offline backend video", "video/x-msvideo")
    data = {"target_state": json.dumps(TARGET), "render_meta": json.dumps(META)}
    with patch.object(web, "OUTPUT", tmp_path), TestClient(web.app) as client:
        assert client.get("/").status_code == 200
        assert client.post("/api/analyze", files={"video": files["video"]}).status_code == 422
        response = client.post("/api/analyze", files=files, data=data)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["result"]["phase"] == "reached"
        for url in body["files"].values():
            assert client.get(url).status_code == 200
        task = tmp_path / body["task_id"]
        assert len(list(task.rglob("*.avi"))) == 1
        assert not list(task.rglob("crop*"))
        assert client.get("/api/runs/not-a-run/result.json").status_code == 404
        assert client.get(f"/api/runs/{body['task_id']}/%2e%2e%2fsecret").status_code == 404
        assert client.post("/api/analyze", files=files, data={**data, "target_state": "{}"}).status_code == 400


def test_vlm_url_and_three_images():
    value = {"phase": "navigation", "reference_reached": False, "target_reached": False,
             "actions": [{"actor": "camera", "action": "pan_left"}], "confidence": .6, "guidance": "向左调整"}
    envelope = {"choices": [{"message": {"content": "```json\n" + json.dumps(value) + "\n```"}}]}
    backend = VLMBackend(endpoint="http://localhost:9999/v1/chat/completions", model="test", api_key="test-key")
    with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(envelope).encode())) as send:
        image = Image.new("RGB", (320, 240))
        result = backend.analyze(image, image, image, Path(__file__), TARGET, META, video_url="https://example.org/original.mp4")
    assert result.phase == "navigation"
    body = json.loads(send.call_args.args[0].data)
    content = body["messages"][0]["content"]
    assert sum(item["type"] == "image_url" for item in content) == 3
    assert body["model"] == "test" and body["enable_thinking"] is False
    assert send.call_args.kwargs["timeout"] == 120
    assert [item for item in content if item["type"] == "video_url"] == [{"type": "video_url", "video_url": {"url": "https://example.org/original.mp4"}, "fps": 2.0}]
