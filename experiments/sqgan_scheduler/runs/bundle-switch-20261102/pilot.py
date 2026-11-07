from pathlib import Path
import sys,importlib.util,json,time,os,hashlib
root=Path.cwd();sys.path.insert(0,str(root))
out=root/"experiments/sqgan_scheduler/runs/bundle-switch-20261102";out.mkdir(parents=True,exist_ok=True)
from Codec.integration.sqgan_payload import SQGANDigitalCodec
import torch
torch.set_num_threads(2)
print("BEGIN verified fine-tuned SQ-GAN loading",flush=True);start=time.perf_counter()
codec=SQGANDigitalCodec()
print("LOADED",time.perf_counter()-start,flush=True)
(out/"load_validation.json").write_text(json.dumps(codec.validation,indent=2))
# Real cached image, independently of scheduler policy order.
import tarfile,numpy as np
from PIL import Image
archive=root/"experiments/mini_dataset/runs/mini_spi_20261030_v1/downloads/Mini_dataset_6000.tar"
with tarfile.open(archive) as tar:
 index=json.load(tar.extractfile("Mini_dataset/portable_image_index.json"));rec=index[0]
 with Image.open(tar.extractfile("Mini_dataset/"+rec["relative_path"])) as opened:
  image=opened.convert("RGB");image.thumbnail((512,256))
  canvas=Image.new("RGB",(512,256),(127,127,127));canvas.paste(image,((512-image.width)//2,(256-image.height)//2))
canvas.save(out/"input.png")
x=torch.from_numpy(np.asarray(canvas).copy()).permute(2,0,1).float()[None]/255
print("BEGIN transmitter semantic teacher",flush=True);ts=time.perf_counter();semantic=codec.semantic_map(x);teacher_seconds=time.perf_counter()-ts
print("BEGIN image -> digital packet",flush=True);ts=time.perf_counter();payload=codec.encode(x,semantic);enc=time.perf_counter()-ts
(out/"message.sq01").write_bytes(payload)
print("PAYLOAD",len(payload),"bytes; BEGIN decode from bytes only",flush=True)
ts=time.perf_counter();decoded=codec.decode(bytes(payload));dec=time.perf_counter()-ts
pixels=decoded[0].clamp(0,1).permute(1,2,0).numpy()
Image.fromarray(np.round(pixels*255).astype(np.uint8)).save(out/"reconstruction.png")
bad=bytearray(payload);bad[-1]^=1
try:codec.decode(bytes(bad))
except ValueError as e:corruption=str(e)
else:raise AssertionError("corrupted packet accepted")
report={"codec":"sqgan","actual_fine_tuned_checkpoint":codec.checkpoint_sha256,"payload_bytes":len(payload),
 "payload_sha256":hashlib.sha256(payload).hexdigest(),"teacher_seconds":teacher_seconds,"encode_seconds":enc,"decode_seconds":dec,
 "shape":list(decoded.shape),"finite":bool(torch.isfinite(decoded).all()),"corruption_rejection":corruption,
 "scope":"real image, trained weights, CPU encode+digital serialization+decode; no radio or quality/generalization claim"}
(out/"pilot.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

