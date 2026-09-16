import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image,ImageDraw
from pydantic import ValidationError
from experiments.fit_target_state.config import FitConfig
from experiments.fit_target_state.geometry import decode
from experiments.fit_target_state.images import ignore_mask,bbox_mask,letterbox,map_box,frame_for
from experiments.fit_target_state.loss import LossEvaluator
from experiments.fit_target_state.run import run_fit,detect_ground_truth,main
from photography_viewpoint_agent.schemas.subject import SubjectObservation
from photography_viewpoint_agent.renderer.target_sketch import TargetSketchRenderer
from photography_viewpoint_agent.schemas.target import TargetState


def textured_image(size=(160,96)):
    image=Image.new('RGB',size,(40,80,130))
    draw=ImageDraw.Draw(image)
    for x in range(0,size[0],20):
        draw.rectangle((x,0,x+7,size[1]),fill=(100+x%155,170,35))
    draw.ellipse((55,15,95,55),fill=(240,60,90))
    draw.rectangle((20,70,130,82),fill=(20,210,230))
    return image


def test_parameterization_preserves_physical_ratio():
    config=FitConfig()
    _,target=decode([0.6,0.4,np.log(2),0.8,0.9,0.4],(1920,1080),(800,800),(0.4,0.2,0.5,0.8),config)
    v=target.framing.reference_viewport
    assert (v[2]-v[0])*1920/((v[3]-v[1])*1080)==pytest.approx(1)
    b=target.subject.bbox
    assert (b[2]-b[0])/(b[3]-b[1])==pytest.approx(192/648)


@pytest.mark.parametrize('data',[{'population_size':0},{'output_width':100},
    {'zoom_bounds':[2,1]},{'weights':{'subject':0,'background':0,'global':0}}])
def test_bad_config(data):
    with pytest.raises(ValidationError): FitConfig.model_validate(data)


def test_ignore_value_one_rectangles_and_outside_bbox(tmp_path):
    mask=np.zeros((20,30),dtype=np.uint8)
    mask[:5,:5]=1
    Image.fromarray(mask).save(tmp_path/'ignore.png')
    combined=np.array(ignore_mask((30,20),tmp_path/'ignore.png',[(0.5,0.5,1,1)]))
    assert combined[0,0]==255 and combined[19,29]==255 and combined[8,8]==0
    assert not bbox_mask((-0.5,0,-0.2,1),(30,20)).any()
    assert not bbox_mask((1.2,0,1.5,1),(30,20)).any()
    with pytest.raises(ValueError):ignore_mask((20,20),tmp_path/'ignore.png',[])


def test_letterbox_bbox_mapping():
    image,mapping=letterbox(Image.new('RGB',(200,100)),(100,100))
    assert map_box((0,0,1,1),(200,100),(100,100),mapping)==(0,0.25,1,0.75)


def test_ignored_text_does_not_affect_loss():
    gt=Image.new('RGB',(80,60),(100,100,100))
    changed=gt.copy()
    ImageDraw.Draw(changed).rectangle((0,0,79,10),fill='white')
    ignore=Image.fromarray(bbox_mask((0,0,1,0.25),gt.size))
    config=FitConfig(subject_mode='follow_reference')
    evaluator=LossEvaluator(gt,None,ignore,config)
    loss,_=evaluator.evaluate(changed,None,np.zeros((60,80),dtype=np.uint8))
    assert loss['total']==0
    with pytest.raises(ValueError):LossEvaluator(gt,None,Image.new('L',gt.size,255),config)


def test_hiding_background_cannot_win():
    image=Image.new('RGB',(80,60),(100,100,100))
    evaluator=LossEvaluator(image,None,Image.new('L',image.size,0),FitConfig(subject_mode='follow_reference'))
    loss,valid=evaluator.evaluate(image,None,np.full((60,80),255,dtype=np.uint8))
    assert not valid.any()
    assert loss['insufficient_background_penalty']==10 and loss['total']>=10


def test_gt_detection_fallback():
    class FailedDetector:
        def detect(self,*args):raise ValueError('no person')
    with pytest.raises(ValueError,match='gt_subject_bbox'):detect_ground_truth(FailedDetector(),None,None,None)
    bbox,info=detect_ground_truth(FailedDetector(),None,None,(0.2,0.2,0.4,0.8))
    assert bbox==(0.2,0.2,0.4,0.8) and info['source']=='manual_fallback'


def test_follow_fit_reproducible_and_default_canvas(tmp_path):
    reference=textured_image()
    ref=tmp_path/'reference.png'
    reference.save(ref)
    target=TargetState.model_validate({'subject':{'mode':'follow_reference'},
        'framing':{'reference_viewport':[0.1,0.1,0.9,0.9]}})
    gt=tmp_path/'gt.png'
    TargetSketchRenderer(160,96).render(frame_for(ref,reference.size),None,target,gt)
    config=FitConfig(subject_mode='follow_reference',population_size=20,num_rounds=4,samples_per_round=20,preview_max_side=160)
    logs=[run_fit(ref,gt,tmp_path/str(i),config,progress=lambda _:None) for i in range(2)]
    assert logs[0]['best_target_state']==logs[1]['best_target_state']
    assert logs[0]['best_loss']==logs[1]['best_loss']
    assert logs[0]['best_loss']['total']<logs[0]['evaluations'][0]['loss']['total']*0.5
    assert Image.open(tmp_path/'0/best_sketch.jpg').size==reference.size
    for name in ['best_target_state.json','comparison_triptych.jpg','search_log.json','debug/masked_gt.jpg','debug/masked_sketch.jpg']:
        assert (tmp_path/'0'/name).is_file()
    with pytest.raises(ValueError,match='empty'):run_fit(ref,gt,tmp_path/'0',config)


def test_reposition_fit_with_offline_detector_and_fallback(tmp_path):
    reference=textured_image()
    ImageDraw.Draw(reference).rectangle((70,25,89,74),fill=(240,0,0))
    ref=tmp_path/'reference.png'
    reference.save(ref)
    mask=np.zeros((96,160),dtype=np.uint8)
    mask[25:75,70:90]=255
    Image.fromarray(mask).save(tmp_path/'mask.png')
    observation=SubjectObservation(bbox=(70/160,25/96,90/160,75/96),mask_path=str(tmp_path/'mask.png'),confidence=0.9)
    gt=tmp_path/'gt.png'
    target=TargetState.model_validate({'subject':{'mode':'reposition','bbox':[0.1,0.2,0.25,0.8]},
        'framing':{'reference_viewport':[0,0,1,1]}})
    _,meta=TargetSketchRenderer(160,96).render(frame_for(ref,reference.size),observation,target,gt)
    class Detector:
        def detect(self,frame,path):
            if frame.frame_id=='gt_native':raise ValueError('GT unavailable')
            return observation
    config=FitConfig(population_size=4,num_rounds=1,samples_per_round=4,
        gt_subject_bbox=meta.rendered_subject_bbox,preview_max_side=160)
    log=run_fit(ref,gt,tmp_path/'out',config,detector=Detector(),progress=lambda _:None)
    assert log['detections']['ground_truth']['source']=='manual_fallback'
    assert log['best_target_state']['subject']['mode']=='reposition'
    assert log['exported_sketch_loss']['subject']<0.02


def test_cli_config_override(tmp_path):
    path=tmp_path/'input.png'
    textured_image().save(path)
    config=tmp_path/'case.json'
    config.write_text(json.dumps({'subject_mode':'reposition','population_size':1,'num_rounds':1,'samples_per_round':1}))
    assert main(['--reference',str(path),'--ground-truth',str(path),'--config',str(config),
                 '--mode','follow_reference','--output-dir',str(tmp_path/'out')])==0
