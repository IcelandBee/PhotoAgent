"""NVIDIA server acceptance: six complete app.py runs, then same-depth A/B."""
import argparse,json,subprocess,sys,time
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from photography_viewpoint_agent.renderer.mesh_warp import PyTorch3DRasterizer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--depth-device',default='cuda')
    p.add_argument('--mesh-device',default='cuda')
    p.add_argument('--depth-pro-model',default='apple/DepthPro-hf')
    args=p.parse_args()
    PyTorch3DRasterizer(args.mesh_device).check_available()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise ValueError('Output directory must be empty')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    repo=Path(__file__).resolve().parents[1]
    cases=[('none','00_baseline__full_frame_follow'),('rotation','40_viewpoint_only__yaw_right_5deg'),
        ('point','51_depth3d__translate_right_small'),('mesh_right','61_depthmesh__translate_right_small'),
        ('mesh_forward','62_depthmesh__translate_forward_small'),('mesh_combined','63_combined__depthmesh__zoom_in__subject_right')]
    records=[]
    for name,example in cases:
        out=args.output_dir/name
        cmd=[sys.executable,str(repo/'app.py'),'--video',args.video,'--target-state',str(repo/'examples'/(example+'.json')),
             '--work-dir',str(out),'--debug','--depth-device',args.depth_device,'--mesh-device',args.mesh_device]
        if name=='mesh_right':cmd+=['--depth-backend','depth_pro','--depth-model',args.depth_pro_model]
        if name in ('mesh_forward','mesh_combined'):
            cmd+=['--depth-path',str(args.output_dir/'mesh_right/reference_depth.npy')]
        start=time.perf_counter();result=subprocess.run(cmd,capture_output=True,text=True)
        (args.output_dir/(name+'.log')).write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        records.append({'case':name,'returncode':result.returncode,'seconds':time.perf_counter()-start})
        (args.output_dir/'summary.json').write_text(json.dumps(records,indent=2))
        print(records[-1],flush=True)
        if result.returncode:raise RuntimeError('Failed workflow; see '+name+'.log')
    subprocess.run([sys.executable,str(repo/'tools/compare_point_mesh.py'),'--result',str(args.output_dir/'mesh_right/result.json'),
        '--output-dir',str(args.output_dir/'ab_same_depth'),'--mesh-device',args.mesh_device],check=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
