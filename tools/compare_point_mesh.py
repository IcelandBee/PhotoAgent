"""Debug-only A/B renderer from one saved workflow result, with identical inputs."""
import argparse
import json
from pathlib import Path
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image,ImageDraw
from photography_viewpoint_agent.renderer.target_sketch import TargetSketchRenderer
from photography_viewpoint_agent.schemas.target import TargetState
from photography_viewpoint_agent.schemas.depth import DepthObservation
from photography_viewpoint_agent.schemas.subject import SubjectObservation
from photography_viewpoint_agent.schemas.video import FrameInfo


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,required=True,help='result.json from a successful depth workflow, on this host')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--mesh-device',default='cuda')
    parser.add_argument('--mesh-stride',type=int,default=2)
    parser.add_argument('--mesh-depth-edge-threshold',type=float,default=.12)
    args=parser.parse_args(argv)
    data=json.loads(args.result.read_text(encoding='utf-8-sig'))
    if data.get('error'):raise ValueError('Input workflow result contains an error')
    frame=FrameInfo.model_validate(data['reference_frame'])
    depth=DepthObservation.model_validate(data['reference_depth'])
    subject=SubjectObservation.model_validate(data['reference_subject']) if data.get('reference_subject') else None
    target=TargetState.model_validate(data['target_state'])
    with Image.open(data['target_sketch_path']) as im:size=im.size
    if not target.viewpoint.active:
        raise ValueError('A/B rotation comparison requires nonzero yaw/pitch/roll')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise ValueError('Output directory must be empty')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    records={}
    for name,mode in [('point','depth_3d'),('mesh','depth_mesh')]:
        renderer=TargetSketchRenderer(*size,debug=True,viewpoint_backend=mode,mesh_device=args.mesh_device,
            mesh_stride=args.mesh_stride,mesh_depth_edge_threshold=args.mesh_depth_edge_threshold)
        renderer.preflight(mode)
        state=target
        path,meta=renderer.render(frame,subject,state,args.output_dir/name/'target_sketch.jpg',reference_depth=depth)
        records[name]={'path':path,'render_meta':meta.model_dump(),'target_state':state.model_dump()}
    panel_w=640;panel_h=round(panel_w*size[1]/size[0]);header=40
    sheet=Image.new('RGB',(panel_w*3,panel_h+header),(25,28,32));draw=ImageDraw.Draw(sheet)
    for col,name in enumerate(['point','mesh']):
        with Image.open(records[name]['path']) as im:sheet.paste(im.resize((panel_w,panel_h)),(col*panel_w,header))
        draw.text((col*panel_w+12,12),name.upper()+' (same reference/depth/K/motion)',fill='white')
    draw.text((2*panel_w+12,12),'Coverage before viewport: POINT top / MESH bottom',fill='white')
    for row,name in enumerate(['point','mesh']):
        with Image.open(args.output_dir/name/'viewpoint_valid_mask.png') as im:
            sheet.paste(im.convert('RGB').resize((panel_w,panel_h//2),Image.Resampling.NEAREST),(2*panel_w,header+row*(panel_h//2)))
    sheet.save(args.output_dir/'point_vs_mesh.jpg',quality=95)
    (args.output_dir/'comparison.json').write_text(json.dumps({'reference':frame.model_dump(),
        'depth':depth.model_dump(),'runs':records,'note':'Same depth and intrinsics isolates representation; not a Depth Anything vs Depth Pro model comparison.'},indent=2),encoding='utf-8')
    print((args.output_dir/'point_vs_mesh.jpg').resolve())
    return 0


if __name__=='__main__':raise SystemExit(main())
