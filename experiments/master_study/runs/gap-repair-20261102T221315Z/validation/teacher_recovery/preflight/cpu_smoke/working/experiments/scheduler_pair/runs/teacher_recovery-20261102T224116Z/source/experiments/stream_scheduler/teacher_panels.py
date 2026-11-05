"""Deterministic visual bindings for offline teacher queries; never actor features."""
import hashlib,json,math,re
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,ImageOps

PANEL_VERSION="labeled_teacher_panels_v1"
PIXEL_BUDGET=2048*28*28

def _font(size):
    for name in ("DejaVuSans.ttf","Arial.ttf"):
        try:return ImageFont.truetype(name,size)
        except OSError:pass
    return ImageFont.load_default()

def _put(canvas,path,box,label):
    draw=ImageDraw.Draw(canvas);x,y,w,h=box
    draw.rectangle((x,y,x+w-1,y+h-1),outline="#555555",width=2)
    font=_font(17)
    while label and draw.textbbox((0,0),label,font=font)[2]>w-16:label=label[:-1]
    draw.text((x+8,y+5),label,fill="black",font=font)
    with Image.open(path) as im:
        rgb=ImageOps.exif_transpose(im).convert("RGB")
        fitted=ImageOps.contain(rgb,(w-12,h-36))
        canvas.paste(fitted,(x+(w-fitted.width)//2,y+30+(h-36-fitted.height)//2))

def render_panel(images,roles,output,kind):
    """All original images remain visible, letterboxed; labels carry no priority."""
    if len(images)!=len(roles) or not 1<=len(images)<=10:raise ValueError("invalid teacher panel membership")
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    originals=[{"path":str(Path(p).resolve()),"sha256":hashlib.sha256(Path(p).read_bytes()).hexdigest(),"role":r}
               for p,r in zip(images,roles)]
    if kind=="candidate":
        current={};recent={"A":[],"B":[]};processed=[]
        for path,role in zip(images,roles):
            match=re.match(r"Candidate ([AB]);",role)
            if match:
                label=match.group(1)
                if label in current:raise ValueError("duplicate CURRENT panel")
                current[label]=(path,role);continue
            match=re.match(r"Source-history-window observation for ([AB]);",role)
            if match:recent[match.group(1)].append((path,role));continue
            if role.startswith("Previously processed source evidence;"):processed.append((path,role));continue
            raise ValueError("unrecognized candidate image role: "+role)
        if set(current)!={"A","B"}:raise ValueError("both CURRENT panels required")
        height=48+384+32+(164 if any(recent.values()) else 40)+(32+180 if processed else 0)
        canvas=Image.new("RGB",(1024,height),"#f4f4f4");draw=ImageDraw.Draw(canvas)
        for j,label in enumerate(("A","B")):
            draw.text((j*512+12,10),"CURRENT "+label+" — eligible next selection",fill="black",font=_font(20))
            _put(canvas,current[label][0],(j*512,48,512,384),"CURRENT "+label)
        y=48+384
        draw.text((10,y+5),"RECENT UE OBSERVATIONS available by decision — not necessarily delivered",fill="black",font=_font(17));y+=32
        for j,label in enumerate(("A","B")):
            items=recent[label]
            if not items:
                draw.text((j*512+10,y+8),"No available recent window for "+label,fill="black",font=_font(17))
            else:
                w=512//len(items)
                for k,(path,role) in enumerate(items):
                    status=role.split("status=",1)[-1]
                    _put(canvas,path,(j*512+k*w,y,w,164),label+" H"+str(k+1)+": "+status)
        y+=164 if any(recent.values()) else 40
        if processed:
            draw.text((10,y+5),"SUCCESSFULLY PROCESSED source evidence — reconstruction not verified",fill="black",font=_font(17));y+=32
            w=1024//len(processed)
            for k,(path,role) in enumerate(processed):_put(canvas,path,(k*w,y,w,180),"Receiver source history "+str(k+1))
        instruction="One labeled contact sheet. Only the two large CURRENT A/B panels are choices. Smaller recent/receiver panels are context. "+" | ".join(roles)
    elif kind in ("prefix","replay"):
        columns=2;tile_width=512;tile_height=352
        canvas=Image.new("RGB",(1024,40+tile_height*math.ceil(len(images)/columns)),"#f4f4f4")
        ImageDraw.Draw(canvas).text((10,8),"EVIDENCE IDs match the schedule text; no implied priority/order",fill="black",font=_font(19))
        for i,(path,role) in enumerate(zip(images,roles)):
            match=re.match(r"Evidence ([^;]+);",role)
            if not match:raise ValueError("missing neutral Evidence ID")
            _put(canvas,path,((i%2)*tile_width,40+(i//2)*tile_height,tile_width,tile_height),"Evidence "+match.group(1))
        instruction="One contact sheet with visible neutral Evidence IDs. Use those IDs to interpret schedules and the following causal metadata. "+" | ".join(roles)
    else:raise ValueError("unrecognized panel kind")
    canvas.save(output)
    receipt={"version":PANEL_VERSION,"kind":kind,"canvas_size":list(canvas.size),"sources":originals,
             "panel_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),"max_pixels":PIXEL_BUDGET,
             "geometry":"aspect-preserving letterbox; no scene content synthesized"}
    output.with_suffix(".json").write_text(json.dumps(receipt,indent=2))
    return {"images":[str(output.resolve())],"roles":[instruction],"visual_layout":PANEL_VERSION,
            "source_images":originals,"pixel_budget":PIXEL_BUDGET}
