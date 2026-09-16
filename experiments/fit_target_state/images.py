from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from photography_viewpoint_agent.schemas.video import FrameInfo


def read_image(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def frame_for(path, size):
    return FrameInfo(frame_id=Path(path).stem,path=str(Path(path).resolve()),frame_index=0,
                     timestamp=0,width=size[0],height=size[1])


def bbox_mask(box, size):
    mask = np.zeros((size[1],size[0]),dtype=np.uint8)
    if box is not None:
        x0,y0,x1,y1 = box
        left,right = np.clip([math.floor(x0*size[0]),math.ceil(x1*size[0])],0,size[0])
        top,bottom = np.clip([math.floor(y0*size[1]),math.ceil(y1*size[1])],0,size[1])
        mask[top:bottom,left:right] = 255
    return mask


def letterbox(image, size, resample=Image.Resampling.BICUBIC, fill=128):
    """One uniform scale; return affine mapping in native pixels -> canvas pixels."""
    scale = min(size[0]/image.width,size[1]/image.height)
    left,top = (size[0]-image.width*scale)/2,(size[1]-image.height*scale)/2
    fillcolor = (fill,fill,fill) if image.mode == "RGB" else fill
    result = image.transform(size,Image.Transform.AFFINE,
        (1/scale,0,-left/scale,0,1/scale,-top/scale),resample,fillcolor=fillcolor)
    return result,(scale,left,top)


def map_box(box, native_size, canvas_size, mapping):
    if box is None:
        return None
    s,left,top = mapping
    return ((box[0]*native_size[0]*s+left)/canvas_size[0],
            (box[1]*native_size[1]*s+top)/canvas_size[1],
            (box[2]*native_size[0]*s+left)/canvas_size[0],
            (box[3]*native_size[1]*s+top)/canvas_size[1])


def ignore_mask(native_size, path, rectangles):
    mask = np.zeros((native_size[1],native_size[0]),dtype=np.uint8)
    if path is not None:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("L")
            if image.size != native_size:
                raise ValueError("ignore-mask must match the oriented ground-truth image dimensions")
            mask[np.array(image)>0] = 255  # Both value 1 and value 255 mean ignored.
    for box in rectangles:
        mask |= bbox_mask(box,native_size)
    return Image.fromarray(mask)


def save_triptych(reference,sketch,gt,path):
    panel_size = (min(640,sketch.width),max(1,round(min(640,sketch.width)*sketch.height/sketch.width)))
    canvas = Image.new("RGB",(panel_size[0]*3,panel_size[1]+36),(35,35,35))
    draw = ImageDraw.Draw(canvas)
    for index,(title,image) in enumerate(zip(("REFERENCE","BEST SKETCH","GROUND TRUTH"),(reference,sketch,gt))):
        panel,_ = letterbox(image,panel_size)
        canvas.paste(panel,(index*panel_size[0],36))
        draw.text((index*panel_size[0]+12,10),title,fill="white")
    canvas.save(path,quality=95)
