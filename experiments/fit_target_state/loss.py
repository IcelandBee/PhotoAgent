import cv2
import numpy as np
from .images import bbox_mask
from .geometry import geometry_features


def gray_and_edges(image):
    gray = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    gray = cv2.GaussianBlur(gray,(3,3),0.6)
    dx = cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)/4
    dy = cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)/4
    return gray,np.clip(np.hypot(dx,dy),0,1)


class LossEvaluator:
    def __init__(self,gt,gt_box,ignore,config):
        self.gt,self.gt_box,self.config = gt,gt_box,config
        self.gray,self.edges = gray_and_edges(gt)
        self.base_valid = ~np.asarray(ignore).astype(bool)
        # Erode validity so ignored captions cannot leak through Gaussian/Sobel kernels.
        self.base_valid = cv2.erode(self.base_valid.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
        self.gt_person = bbox_mask(gt_box,gt.size)>0
        self.weight_values = np.array([config.weights.subject if gt_box is not None and config.subject_mode=='reposition' else 0,
                                      config.weights.background,config.weights.global_],dtype=float)
        self.weight_values /= self.weight_values.sum()
        if self.base_valid.sum() < 16:
            raise ValueError("Ignore mask leaves too few usable ground-truth pixels")

    def evaluate(self,sketch,subject_bbox,excluded):
        gray,edges = gray_and_edges(sketch)
        subject = (float(np.mean(np.abs(geometry_features(subject_bbox)-geometry_features(self.gt_box))))
                   if self.gt_box is not None and subject_bbox is not None else 0.0)
        excluded = cv2.dilate((excluded>0).astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
        gt_person = cv2.dilate(self.gt_person.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
        valid = self.base_valid & ~gt_person & ~excluded
        fraction = float(valid.sum()/self.base_valid.sum())
        if fraction < self.config.min_background_fraction or valid.sum() < 16:
            background_l1,edge,background = 1.0,1.0,1.0
            penalty = 10.0  # Do not reward candidates that hide almost all comparison pixels.
        else:
            background_l1 = float(np.mean(np.abs(gray[valid]-self.gray[valid])))
            edge = float(np.mean(np.abs(edges[valid]-self.edges[valid])))
            background = 0.7*background_l1+0.3*edge
            penalty = 0.0
        global_loss = float(np.mean(np.abs(gray[self.base_valid]-self.gray[self.base_valid])))
        total = float(np.dot(self.weight_values,[subject,background,global_loss])+penalty)
        return {"total":total,"subject":subject,"background":background,"background_l1":background_l1,
                "background_edges":edge,"global":global_loss,"background_valid_fraction":fraction,
                "insufficient_background_penalty":penalty},valid
