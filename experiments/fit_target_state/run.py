"""Standalone CLI; run from any directory without changing the production workflow."""
import argparse
import json
import logging
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image
from experiments.fit_target_state.config import FitConfig
from experiments.fit_target_state.images import read_image,frame_for,letterbox,map_box,ignore_mask,bbox_mask,save_triptych
from experiments.fit_target_state.preview import PreparedRenderer
from experiments.fit_target_state.loss import LossEvaluator
from experiments.fit_target_state.search import search
from photography_viewpoint_agent.subject_detection.yolo import YOLOSubjectDetector
from photography_viewpoint_agent.renderer.target_sketch import TargetSketchRenderer


def detect_ground_truth(detector,frame,path,fallback):
    try:
        observation = detector.detect(frame,path)
        return observation.bbox,{"source":"yolo","observation":observation.model_dump()}
    except Exception as exc:
        if fallback is None:
            raise ValueError(f"Ground-truth person detection failed: {exc}. Provide gt_subject_bbox in --config as fallback.") from exc
        return fallback,{"source":"manual_fallback","detection_error":str(exc),"bbox":fallback}


def run_fit(reference_path,gt_path,output_dir,config,ignore_path=None,detector=None,progress=print):
    started = time.perf_counter()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory must be empty: {output_dir}")
    reference,gt_native = read_image(reference_path),read_image(gt_path)
    native_ignore = ignore_mask(gt_native.size,ignore_path,config.ignore_rects)
    output_dir.mkdir(parents=True,exist_ok=True)
    debug = output_dir/'debug'
    debug.mkdir()
    reference.save(debug/'reference.png')
    gt_native.save(debug/'gt_native.png')
    reference_frame = frame_for(debug/'reference.png',reference.size)
    canvas_size = ((config.output_width,config.output_height) if config.output_width else reference.size)
    gt,mapping = letterbox(gt_native,canvas_size)
    # Outside native GT is ignored, never scored as a framing mismatch.
    ignore,_ = letterbox(native_ignore,canvas_size,Image.Resampling.NEAREST,fill=255)
    observation = None
    gt_box = config.gt_subject_bbox if config.subject_mode=='follow_reference' else None
    detection_log = {"reference":None,"ground_truth":{"source":"not_requested"}}
    if config.subject_mode == 'reposition':
        detector = detector or YOLOSubjectDetector(config.yolo_model,config.person_confidence,config.device)
        progress('Detecting reference and ground-truth subjects...')
        try:
            observation = detector.detect(reference_frame,debug/'reference_mask.png')
        except Exception as exc:
            raise ValueError(f"Reference person detection failed (required for reposition): {exc}") from exc
        detection_log['reference'] = observation.model_dump()
        gt_box,detection_log['ground_truth'] = detect_ground_truth(detector,
            frame_for(debug/'gt_native.png',gt_native.size),debug/'gt_mask.png',config.gt_subject_bbox)
    elif gt_box is not None:
        detection_log['ground_truth'] = {"source":"manual_background_exclusion","bbox":gt_box}
    gt_box = map_box(gt_box,gt_native.size,canvas_size,mapping)
    ratio = min(1,config.preview_max_side/max(canvas_size))
    preview_size = (max(1,round(canvas_size[0]*ratio)),max(1,round(canvas_size[1]*ratio)))
    gt_preview = gt.resize(preview_size,Image.Resampling.LANCZOS)
    ignore_preview = ignore.resize(preview_size,Image.Resampling.NEAREST)
    evaluator = LossEvaluator(gt_preview,gt_box,ignore_preview,config)
    prepared = PreparedRenderer(reference,observation,preview_size)
    best,history,rounds = search(prepared,evaluator,reference.size,canvas_size,
        observation.bbox if observation is not None else None,gt_box,config,progress)
    target = best['target']
    (output_dir/'best_target_state.json').write_text(target.model_dump_json(indent=2),encoding='utf-8')
    progress('Rendering best target at final resolution...')
    renderer = TargetSketchRenderer(*canvas_size,debug=True)
    path,meta = renderer.render(reference_frame,observation,target,output_dir/'best_sketch.jpg')
    sketch = read_image(path)
    # Re-evaluate the actual exported JPEG, so approximation/codec gaps are visible.
    exported_preview = sketch.resize(preview_size,Image.Resampling.LANCZOS)
    exclusion = bbox_mask(meta.rendered_subject_bbox,preview_size)
    if meta.natural_subject_bbox is not None:
        exclusion |= bbox_mask(meta.natural_subject_bbox,preview_size)
    final_loss,valid = evaluator.evaluate(exported_preview,meta.rendered_subject_bbox,exclusion)
    for name,image in [('masked_gt',gt_preview),('masked_sketch',exported_preview)]:
        pixels = np.array(image)
        pixels[~valid] = (128,128,128)
        Image.fromarray(pixels).save(debug/f'{name}.jpg',quality=95)
    Image.fromarray(valid.astype(np.uint8)*255).save(debug/'background_valid_mask.png')
    ignore.save(debug/'effective_ignore_mask.png')
    save_triptych(reference,sketch,gt_native,output_dir/'comparison_triptych.jpg')
    log = {
        "config":config.model_dump(mode='json',by_alias=True),
        "inputs":{"reference":str(Path(reference_path).resolve()),"ground_truth":str(Path(gt_path).resolve()),
                  "ignore_mask":str(Path(ignore_path).resolve()) if ignore_path else None},
        "reference_size":reference.size,"gt_native_size":gt_native.size,"canvas_size":canvas_size,
        "preview_size":preview_size,"gt_to_canvas":{"scale":mapping[0],"left":mapping[1],"top":mapping[2]},
        "detections":detection_log,"gt_subject_bbox_on_canvas":gt_box,
        "loss_definition":{
            "subject":"mean absolute difference of normalized bottom_center_x, bottom_y, height",
            "background":"0.7 * grayscale_L1 + 0.3 * Sobel_magnitude_L1 on valid background",
            "global":"grayscale_L1 on GT-valid, non-ignored pixels (persons included)",
            "mask":"GT person bbox + rendered person bbox + projected original person bbox + ignore mask/rects + GT letterbox; 3px preview safety margin",
            "weights_normalized":dict(zip(('subject','background','global'),evaluator.weight_values.tolist())),
            "penalty":"+10 when valid background fraction < min_background_fraction or fewer than 16 pixels",
        },
        "best_iteration":best['iteration'],"best_loss":best['loss'],"exported_sketch_loss":final_loss,
        "best_parameters":history[best['iteration']]['parameters'],"best_target_state":target.model_dump(),
        "render_meta":meta.model_dump(),"rounds":rounds,"evaluations":history,
        "elapsed_seconds":time.perf_counter()-started,
        "limitations":["Synthetic 2D similarity model cannot reproduce perspective, pose, parallax or invisible scenery.",
                        "Preview and exported JPEG losses can differ due to raster sampling.",
                        "Selected-person bboxes exclude the main subject; other moving people need ignore masks."]}
    (output_dir/'search_log.json').write_text(json.dumps(log,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return log


def main(argv=None):
    parser = argparse.ArgumentParser(description='Fit an experimental TargetState to a static before/after image pair')
    parser.add_argument('--reference',required=True,type=Path)
    parser.add_argument('--ground-truth',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--ignore-mask',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--device')
    parser.add_argument('--mode',choices=['follow_reference','reposition'])
    parser.add_argument('--output-width',type=int)
    parser.add_argument('--output-height',type=int)
    parser.add_argument('--seed',type=int)
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.config.read_text(encoding='utf-8-sig')) if args.config else {}
        for field in ('device','output_width','output_height','seed'):
            if getattr(args,field) is not None:
                data[field] = getattr(args,field)
        if args.mode is not None:
            data['subject_mode'] = args.mode
        config = FitConfig.model_validate(data)
        log = run_fit(args.reference,args.ground_truth,args.output_dir,config,args.ignore_mask)
        print(f"Best preview loss: {log['best_loss']['total']:.6f}; exported sketch loss: {log['exported_sketch_loss']['total']:.6f}")
        print(f"Outputs: {args.output_dir.resolve()}")
        return 0
    except Exception:
        logging.exception('TargetState fitting failed')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
