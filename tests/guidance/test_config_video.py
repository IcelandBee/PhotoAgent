import base64
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from video_guide.core.backends import VLMBackend
from .conftest import TARGET, META


class ConfigVideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "vlm.json"
        self.values = dict(endpoint="https://example.org/v1/chat/completions", model="qwen3.8-max", api_key="file-key", fps=3, timeout=100)
        self.config.write_text(json.dumps(self.values))

    def test_config_reload_and_environment_precedence(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(VLMBackend(config_path=self.config).fps, 3)
            self.config.write_text(json.dumps({**self.values, "fps": 4}))
            self.assertEqual(VLMBackend(config_path=self.config).fps, 4)
            with patch.dict(os.environ, {"VLM_FPS": "5", "VLM_API_KEY": "env-key"}):
                backend = VLMBackend(config_path=self.config)
                self.assertEqual(backend.fps, 5)
                self.assertEqual(backend.api_key, "env-key")
                self.assertEqual(VLMBackend(config_path=self.config, fps=6).fps, 6)

    def test_invalid_config_fails_before_network(self):
        with patch.dict(os.environ, {}, clear=True):
            for changes in ({"fps": 0}, {"fps": 11}, {"fps": True}, {"fps": "nan"}, {"timeout": 0}, {"endpoint": "https://example.org/v1"}, {"endpoint": "https://{WorkspaceId}.example.org/chat/completions"}):
                self.config.write_text(json.dumps({**self.values, **changes}))
                with self.assertRaises(ValueError):
                    VLMBackend(config_path=self.config)

    def test_video_bytes_are_not_sampled_or_reencoded(self):
        video = self.root / "original.mp4"
        content = b"original video payload for transport test"
        video.write_bytes(content)
        response = {"choices": [{"message": {"content": json.dumps(dict(phase="uncertain", reference_reached=False, target_reached=False, actions=[], confidence=.5, guidance="复核目标"))}}]}
        with patch.dict(os.environ, {}, clear=True), patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as send:
            backend = VLMBackend(config_path=self.config)
            image = Image.new("RGB", (32, 32))
            result = backend.analyze(image, image, image, video, TARGET, META)
        body = json.loads(send.call_args.args[0].data)
        blocks = body["messages"][0]["content"]
        item = next(item for item in blocks if item["type"] == "video_url")
        self.assertEqual(base64.b64decode(item["video_url"]["url"].split(",")[1]), content)
        self.assertEqual(item["fps"], 3)
        self.assertEqual(sum(b["type"] == "image_url" for b in blocks), 3)
        prompt = blocks[0]["text"]
        self.assertIn(json.dumps(TARGET, ensure_ascii=False), prompt)
        self.assertIn(json.dumps(META, ensure_ascii=False), prompt)
        self.assertIn("follow_reference", prompt)
        self.assertIn("translation_x/y/z", prompt)
        self.assertIn("禁止只根据 viewport shift", prompt)
        self.assertIn("Point Cloud", prompt)
        self.assertNotIn('"subject": [', prompt)
        self.assertEqual(result.evidence["video_transport"], "base64")

    def test_oversized_base64_rejected_before_request(self):
        video = self.root / "large.mp4"
        with video.open("wb") as output:
            output.truncate(75_000_000)
        with patch.dict(os.environ, {}, clear=True), patch("urllib.request.urlopen") as send:
            backend = VLMBackend(config_path=self.config)
            image = Image.new("RGB", (32, 32))
            with self.assertRaisesRegex(ValueError, "100 MB"):
                backend.analyze(image, image, image, video, TARGET, META)
            send.assert_not_called()


    def test_explicit_images_mode_does_not_read_video(self):
        response = {"choices": [{"message": {"content": json.dumps(dict(phase="uncertain", reference_reached=False, target_reached=False, actions=[], confidence=.5, guidance="复核目标"))}}]}
        with patch.dict(os.environ, {}, clear=True), patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as send:
            backend = VLMBackend(config_path=self.config, input_mode="images", reasoning_effort="low")
            image = Image.new("RGB", (32, 32))
            result = backend.analyze(image, image, image, self.root / "not-read.mp4", TARGET, META)
        body = json.loads(send.call_args.args[0].data)
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertNotIn("enable_thinking", body)
        blocks = body["messages"][0]["content"]
        self.assertEqual(sum(b["type"] == "image_url" for b in blocks), 3)
        self.assertFalse(any(b["type"] == "video_url" for b in blocks))
        self.assertIn("不提供视频", blocks[0]["text"])
        self.assertEqual(result.evidence["video_transport"], "none")
        self.assertEqual(result.evidence["input_mode"], "images")
