"""Experiment-only resolution control for small subjects."""
from pathlib import Path
import numpy as np
from PIL import Image
from photography_viewpoint_agent.schemas.subject import SubjectObservation
from photography_viewpoint_agent.subject_detection.yolo import select_main_person


class ExperimentDetector:
    def __init__(self, config):
        from ultralytics import YOLO
        self.model = YOLO(config.yolo_model)
        self.config = config

    def detect(self, frame, mask_path):
        with Image.open(frame.path) as image:
            image = image.convert('RGB')
            width, height = image.size
            ids = [int(i) for i, name in self.model.names.items() if name == 'person']
            if self.model.task != 'segment' or not ids:
                raise ValueError('A person segmentation model is required')
            result = self.model.predict(image, classes=ids, conf=self.config.person_confidence,
                imgsz=self.config.detection_image_size, device=self.config.device,
                retina_masks=True, verbose=False)[0]
        index = select_main_person(result.boxes.xyxy.cpu().numpy(), width, height)
        if result.masks is None:
            raise ValueError('Missing person mask')
        mask = result.masks.data[index].cpu().numpy()
        if mask.shape != (height, width) or not np.isfinite(mask).all():
            raise ValueError('Invalid person mask dimensions or values')
        binary = Image.fromarray((mask > .5).astype(np.uint8)*255)
        bounds = binary.getbbox()
        if bounds is None:
            raise ValueError('Empty person mask')
        path = Path(mask_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        binary.save(path)
        return SubjectObservation(bbox=tuple(v/d for v,d in zip(bounds,(width,height,width,height))),
            mask_path=str(path.resolve()), confidence=float(result.boxes.conf[index]))
