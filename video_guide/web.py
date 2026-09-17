"""本机演示服务：python -m uvicorn video_guide.web:app --host 127.0.0.1"""
import json
import logging
import os
import time
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .core import GuideInput, GuideService
from .core.backends import LocalBackend, VLMBackend

app = FastAPI(title="视频取景指导", docs_url="/docs")
app.mount("/assets", StaticFiles(directory=Path(__file__).parent / "static"), name="assets")
OUTPUT = Path(os.getenv("GUIDE_OUTPUT_DIR", "outputs")).resolve()
MAX_UPLOAD = 200 * 1024 * 1024
MAX_IMAGE = 20 * 1024 * 1024
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html", headers={"Cache-Control": "no-cache"})


def save_upload(upload, destination, limit, message):
    size = 0
    with destination.open("wb") as file:
        while chunk := upload.file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                raise HTTPException(413, message)
            file.write(chunk)


@app.post("/api/analyze")
def analyze(video: UploadFile = File(...), current_frame: UploadFile = File(...),
            reference_frame: UploadFile = File(...), target_sketch: UploadFile = File(...),
            target_state: str = Form(...), render_meta: str = Form(...),
            backend: str = Form("local"), video_url: str = Form("")):
    uploads = (video, current_frame, reference_frame, target_sketch)
    started = time.perf_counter()
    try:
        if backend not in ("local", "vlm"):
            raise ValueError("不支持的分析后端")
        if Path(video.filename or "").suffix.lower() not in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
            raise ValueError("不支持的视频文件格式")
        for upload in uploads[1:]:
            if Path(upload.filename or "").suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError("请上传有效图片格式")
        target_data, render_data = json.loads(target_state), json.loads(render_meta)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        task = OUTPUT / uuid.uuid4().hex
        source = task / "source"
        source.mkdir(parents=True)
        paths = {}
        for name, upload in zip(("video_path", "current_frame", "reference_frame", "target_sketch"), uploads):
            destination = source / (name + Path(upload.filename).suffix.lower())
            save_upload(upload, destination, MAX_UPLOAD if name == "video_path" else MAX_IMAGE, "上传文件过大")
            paths[name] = destination
        engine = LocalBackend() if backend == "local" else VLMBackend()
        result, folder = GuideService(engine).run(
            GuideInput(**paths, target_state=target_data, render_meta=render_data,
                       session_id=task.name, video_url=video_url or None), task)
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        files = {key: f"/api/runs/{task.name}/{Path(path).relative_to(task).as_posix()}"
                 for key, path in manifest["inputs"].items() if key != "video_path"}
        files["result"] = f"/api/runs/{task.name}/{folder.relative_to(task).as_posix()}/result.json"
        # The video lives once at session/source; image uploads have been archived.
        for name in ("current_frame", "reference_frame", "target_sketch"):
            paths[name].unlink()
        return {"task_id": task.name, "result": result.to_dict(), "files": files,
                "timing": {"total_seconds": round(time.perf_counter() - started, 2)}}
    except HTTPException:
        raise
    except (ValueError, TypeError) as error:
        raise HTTPException(400, str(error)) from error
    except Exception as error:
        logging.getLogger(__name__).exception("分析失败")
        raise HTTPException(502, "视频解码或模型服务调用失败，请检查日志。") from error
    finally:
        for upload in uploads:
            upload.file.close()


@app.get("/api/runs/{run_id}/{filename:path}")
def artifact(run_id: str, filename: str):
    if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
        raise HTTPException(404)
    folder = OUTPUT / run_id
    target = (folder / filename).resolve()
    if not target.is_relative_to(folder.resolve()) or not target.is_file():
        raise HTTPException(404)
    return FileResponse(target)
