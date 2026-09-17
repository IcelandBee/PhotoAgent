"""Optional HF Depth Pro adapter. Raw output is inverse depth: post-process once."""
import time
import math
from PIL import Image
from .storage import save_observation


class DepthProEstimator:
    def __init__(self,model='apple/DepthPro-hf',device='cpu'):
        self.model_name,self.device=model,device
        self.model=self.processor=None

    def estimate(self,frame,output_path):
        import torch
        try:
            from transformers import DepthProImageProcessor,DepthProForDepthEstimation
        except ImportError as exc:
            raise RuntimeError('Depth Pro requires the optional requirements-mesh.txt (Transformers 4.57.6 tested)') from exc
        started=time.perf_counter()
        if self.model is None:
            self.processor=DepthProImageProcessor.from_pretrained(self.model_name)
            self.model=DepthProForDepthEstimation.from_pretrained(self.model_name,
                use_fov_model=True,use_safetensors=True).to(self.device).eval()
        load_seconds=time.perf_counter()-started
        with Image.open(frame.path) as image:
            inputs=self.processor(images=image.convert('RGB'),return_tensors='pt').to(self.device)
        if str(self.device).startswith('cuda'):torch.cuda.synchronize(self.device)
        inference_started=time.perf_counter()
        with torch.inference_mode():
            outputs=self.model(**inputs)
            result=self.processor.post_process_depth_estimation(outputs,
                target_sizes=[(frame.height,frame.width)])[0]
        if str(self.device).startswith('cuda'):torch.cuda.synchronize(self.device)
        inference_seconds=time.perf_counter()-inference_started
        def scalar(key):
            value=result.get(key)
            return float(value.detach().cpu().item()) if value is not None else None
        focal,fov=scalar('focal_length'),scalar('field_of_view')
        if focal is None or not math.isfinite(focal) or focal<=0:
            raise ValueError('Depth Pro did not return a valid calibrated focal length')
        if fov is not None and (not math.isfinite(fov) or not 0<fov<180):
            raise ValueError('Depth Pro returned invalid FoV')
        depth=result['predicted_depth'].detach().float().cpu().numpy()
        return save_observation(depth,frame,output_path,{'source':'depth_pro','depth_backend':'depth_pro',
            'model':self.model_name,'device':self.device,'focal_length_px':focal,
            'horizontal_fov_deg':fov,'focal_reference_width':frame.width,
            'model_load_seconds':load_seconds,'depth_inference_seconds':inference_seconds,
            'estimator_total_seconds':time.perf_counter()-started,
            'conversion':'HF post_process_depth_estimation with original target size; positive metric Z, no normalization'},metric=True)
