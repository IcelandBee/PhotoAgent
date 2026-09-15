from pathlib import Path
import numpy as np
from PIL import Image
from photography_viewpoint_agent.schemas.subject import SubjectObservation
from photography_viewpoint_agent.schemas.video import FrameInfo


def select_main_person(boxes: np.ndarray, width: int, height: int) -> int:
    """Largest area; among candidates within 10% of largest, prefer image center."""
    if len(boxes) == 0:
        raise ValueError("YOLO-seg detected no person")
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("YOLO-seg returned invalid person bbox")
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    eligible = np.flatnonzero(areas >= 0.9 * areas.max())
    centers = (boxes[:, :2] + boxes[:, 2:]) / (2 * np.array([width, height]))
    return int(min(eligible, key=lambda i: (np.linalg.norm(centers[i] - 0.5), -areas[i], i)))


class YOLOSubjectDetector:
    def __init__(self, model_name="yolo11n-seg.pt", confidence=0.4, device="cpu"):
        self.model_name, self.confidence, self.device = model_name, confidence, device
        self._model = None

    def detect(self, frame: FrameInfo, mask_path: Path) -> SubjectObservation:
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_name)
        if self._model.task != "segment":
            raise ValueError("A YOLO segmentation model is required")
        person_ids = [int(i) for i, name in self._model.names.items() if name == "person"]
        if not person_ids:
            raise ValueError("YOLO model has no person class")
        with Image.open(frame.path) as source:
            source = source.convert("RGB")
            width, height = source.size
            result = self._model.predict(source=source, classes=person_ids, conf=self.confidence,
                                         device=self.device, retina_masks=True, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            raise ValueError("YOLO-seg detected no person")
        boxes = result.boxes.xyxy.cpu().numpy()
        index = select_main_person(boxes, width, height)
        if result.masks is None or len(result.masks.data) != len(boxes):
            raise ValueError("YOLO-seg returned missing or mismatched masks")
        mask = result.masks.data[index].cpu().numpy()
        if mask.shape != (height, width) or not np.isfinite(mask).all():
            raise ValueError("YOLO-seg mask must match reference image dimensions and be finite")
        binary = Image.fromarray((mask > 0.5).astype(np.uint8) * 255)
        bounds = binary.getbbox()
        if bounds is None:
            raise ValueError("YOLO-seg returned an empty person mask")
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        binary.save(mask_path)
        return SubjectObservation(bbox=(bounds[0]/width, bounds[1]/height,
                                        bounds[2]/width, bounds[3]/height),
                                  mask_path=str(mask_path.resolve()),
                                  confidence=float(result.boxes.conf[index].item()))
