from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from photography_viewpoint_agent.subject_detection.yolo import YOLOSubjectDetector, select_main_person


def test_select_largest_and_center_tiebreak():
    assert select_main_person(np.array([[0,0,40,80],[45,45,55,55]]),100,100) == 0
    assert select_main_person(np.array([[0,0,20,40],[40,30,60,69]]),100,100) == 1
    with pytest.raises(ValueError,match='no person'):
        select_main_person(np.empty((0,4)),100,100)
    with pytest.raises(ValueError,match='invalid'):
        select_main_person(np.array([[0,0,0,1]]),100,100)


class Tensor:
    def __init__(self, value): self.value = np.asarray(value)
    def cpu(self): return self
    def numpy(self): return self.value
    def item(self): return self.value.item()
    def __getitem__(self,key): return Tensor(self.value[key])
    def __len__(self): return len(self.value)


class Boxes:
    def __init__(self, count):
        self.xyxy = Tensor([[10,10,30,80]]*count)
        self.conf = Tensor([0.9]*count)
    def __len__(self): return len(self.xyxy)


@pytest.mark.parametrize('kind',['valid','no_person','empty_mask','missing_mask','wrong_size','nonfinite'])
def test_detector_adapter_offline(frame,tmp_path,kind):
    mask = np.zeros((100,100),dtype=np.float32)
    mask[10:80,10:30] = 1
    if kind == 'empty_mask': mask[:] = 0
    if kind == 'wrong_size': mask = mask[:50]
    if kind == 'nonfinite': mask[0,0] = np.nan
    result = SimpleNamespace(boxes=Boxes(0 if kind == 'no_person' else 1),
        masks=None if kind == 'missing_mask' else SimpleNamespace(data=Tensor([mask])))
    calls = []
    def predict(**kwargs):
        calls.append(kwargs)
        return [result]
    detector = YOLOSubjectDetector()
    detector._model = SimpleNamespace(task='segment',names={0:'person',1:'car'},predict=predict)
    if kind != 'valid':
        with pytest.raises(ValueError):
            detector.detect(frame,tmp_path/'mask.png')
    else:
        observed = detector.detect(frame,tmp_path/'mask.png')
        assert observed.bbox == (0.1,0.1,0.3,0.8)
        assert Image.open(observed.mask_path).getbbox() == (10,10,30,80)
        assert calls[0]['classes'] == [0] and calls[0]['retina_masks']
        assert calls[0]['device'] == 'cpu'
