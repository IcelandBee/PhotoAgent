"""Standalone deterministic CPU rotation demo and contact sheet."""
import argparse
import json
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from PIL import Image, ImageDraw, ImageOps
from photography_viewpoint_agent.renderer.viewpoint_warp import RotationViewpointWarper


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--yaw',type=float,default=0)
    parser.add_argument('--pitch',type=float,default=0)
    parser.add_argument('--roll',type=float,default=0)
    parser.add_argument('--horizontal-fov',type=float,default=60)
    parser.add_argument('--border-mode',choices=['constant','replicate'],default='constant')
    parser.add_argument('--fill-color',nargs=3,type=int,default=[128,128,128])
    parser.add_argument('--grid',action='store_true',help='Also create viewpoint_grid.jpg beside output')
    args = parser.parse_args(argv)
    if any(v<0 or v>255 for v in args.fill_color):
        parser.error('fill-color channels must be between 0 and 255')
    with Image.open(args.input) as im:
        source = ImageOps.exif_transpose(im).convert('RGB')
    warper = RotationViewpointWarper()
    options = dict(roll_deg=args.roll,horizontal_fov_deg=args.horizontal_fov,
                   border_mode=args.border_mode,fill_color=tuple(args.fill_color))
    image,meta = warper.warp(source,yaw_deg=args.yaw,pitch_deg=args.pitch,**options)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    image.save(args.output,quality=95)
    args.output.with_suffix('.warp.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    if args.grid:
        angles = [-10,-5,0,5,10]
        scale = min(280/source.width,320/source.height,1)
        tw,th = round(source.width*scale),round(source.height*scale)
        sheet = Image.new('RGB',(5*tw,5*(th+28)),(35,35,35))
        draw = ImageDraw.Draw(sheet)
        records=[]
        for row,pitch in enumerate(angles):
            for col,yaw in enumerate(angles):
                result,record = warper.warp(source,yaw_deg=yaw,pitch_deg=pitch,**options)
                result.save(args.output.parent/f'yaw_{yaw:+03d}_pitch_{pitch:+03d}.jpg',quality=95)
                sheet.paste(result.resize((tw,th),Image.Resampling.LANCZOS),(col*tw,row*(th+28)+28))
                draw.text((col*tw+5,row*(th+28)+6),f'yaw {yaw:+d} / pitch {pitch:+d}',fill='white')
                records.append(record)
        sheet.save(args.output.parent/'viewpoint_grid.jpg',quality=95)
        (args.output.parent/'grid_metadata.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(args.output.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
