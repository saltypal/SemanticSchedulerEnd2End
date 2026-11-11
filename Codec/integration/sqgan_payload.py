"""Digital SQ-GAN payload adapter using the user's verified teammate bundle.

SQ01 is the existing project's experimental source packet, not the paper entropy
coder. No radio simulation, retransmission or analog latent transport is added.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.util
import json
import sys

DEFAULT_ROOT=Path(__file__).resolve().parents[3]/"SemanticLossless"/"Semantic_Lossless"
DEFAULT_BUNDLE="Results/team_downloads/2/satyapaladuguplus__semantic-notebook-2/0b680269ce1a4e2c83d9fe5159b3af1c/published_output/models/teammate_2_models/SQ-GAN/20261028T130100Z-eea4f3aa"
COMPONENTS=("encoder","decoder","masker","demasker","quantize","encoder_semantic","decoder_semantic","masker_semantic","demasker_semantic","quantize_semantic")

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def load_source(root):
    root=Path(root).resolve()
    alias="_scheduler_sqgan_"+hashlib.sha256(str(root).encode()).hexdigest()[:12]
    if alias not in sys.modules:
        spec=importlib.util.spec_from_file_location(alias,root/"src/__init__.py",submodule_search_locations=[str(root/"src")])
        module=importlib.util.module_from_spec(spec);sys.modules[alias]=module;spec.loader.exec_module(module)
    return importlib.import_module(alias+".sqgan"),importlib.import_module(alias+".bitstream")

class SQGANDigitalCodec:
    representation="digital_bytes"
    def __init__(self,bundle=None,source_root=None,device="cpu",image_fraction=.25,semantic_fraction=.25):
        import torch
        self.device=str(device);self.source_root=Path(source_root or DEFAULT_ROOT).resolve()
        self.bundle=Path(bundle or self.source_root/DEFAULT_BUNDLE).resolve()
        manifest=json.loads((self.bundle/"bundle.json").read_text())
        if manifest.get("algorithm")!="sqgan" or not manifest.get("pipeline_completed"):
            raise ValueError("completed SQ-GAN bundle required")
        # Verify every packaged file, including fine-tuned last.pt; never silently
        # substitute the constructor components for the trained checkpoint.
        for entry in manifest["files"]:
            path=(self.bundle/entry["path"]).resolve()
            if not path.is_relative_to(self.bundle) or not path.is_file() or path.stat().st_size!=entry["bytes"] or sha256(path)!=entry["sha256"]:
                raise ValueError("SQ-GAN bundle mismatch: "+entry["path"])
        self.checkpoint_sha256=sha256(self.bundle/"last.pt")
        builder,self.packet=load_source(self.source_root)
        self.model=builder.build_sqgan(SimpleNamespace(checkpoint=str(self.bundle/"assets/checkpoint"),device="cpu",
            model_options={"image_fraction":image_fraction,"semantic_fraction":semantic_fraction}))
        saved=torch.load(self.bundle/"last.pt",map_location="cpu",weights_only=True,mmap=True)
        trained=saved["state_dict"];receipts=[]
        for name in COMPONENTS:
            prefix="core."+name+"."
            state={key[len(prefix):]:value for key,value in trained.items() if key.startswith(prefix)}
            if not state:raise ValueError("fine-tuned component missing: "+name)
            getattr(self.model,name).load_state_dict(state,strict=True)
            receipts.append({"component":name,"strict":True,"tensor_count":len(state)})
        # Preserve a memory-mapped reference only for lazy transmitter teacher loading.
        self._trained=trained
        self.model.to(self.device).eval()
        self.validation={"algorithm":"sqgan","checkpoint_sha256":self.checkpoint_sha256,"components":receipts,
            "source_packet":"SQ01","representation":"digital_bytes","source_lossless":False,
            "radio_channel":"none; ideal received payload","input_shape":[1,3,256,512]}
    def encode(self,image,semantic=None):
        import torch
        image=image.to(self.device)
        if image.shape!=(1,3,256,512):raise ValueError('RGB tensor must have shape1x3x256x512')
        if semantic is None:semantic=self.semantic_map(image)
        semantic=semantic.to(self.device)
        if not torch.isfinite(image).all() or torch.any((image<0)|(image>1)):
            raise ValueError("finite RGB input in0..1 required")
        if semantic.shape!=(1,20,256,512) or not torch.isfinite(semantic).all() or not torch.all((semantic==0)|(semantic==1)) or not torch.all(semantic.sum(1)==1):
            raise ValueError("paired one-hot20-class semantic map required; never substitute fabricated masks")
        with torch.inference_mode():payload=self.model.encode(image,semantic)
        if not isinstance(payload,bytes):raise TypeError("SQ-GAN encoder must produce bytes")
        self.packet.decode_packet(payload)
        return payload
    def encode_with_features(self, image, semantic=None):
        """Encode once and return the payload with its pre-mask image grid."""
        from Codec.integration.sqgan_prefeatures import encode_with_features

        return encode_with_features(self, image, semantic)

    def semantic_map(self,image):
        """Transmitter-only frozen teacher restored from the SAME trained bundle."""
        import torch
        from torch.nn import functional as F
        if not hasattr(self,"_teacher"):
            from transformers import AutoConfig,SegformerForSemanticSegmentation
            opts=json.loads((self.bundle/"portable_config.json").read_text())
            cfg=AutoConfig.from_pretrained(opts["teacher_model"],revision=opts["teacher_revision"])
            teacher=SegformerForSemanticSegmentation(cfg)
            state={key[len("teacher."):]:value for key,value in self._trained.items() if key.startswith("teacher.")}
            teacher.load_state_dict(state,strict=True)
            self._teacher=teacher.to(self.device).eval()
        image=image.to(self.device)
        mean=image.new_tensor([.485,.456,.406])[None,:,None,None];std=image.new_tensor([.229,.224,.225])[None,:,None,None]
        with torch.inference_mode():
            logits=self._teacher(pixel_values=(image-mean)/std).logits
            return self._one_hot_from_logits(logits)
    def _seg_index(self):
        """ADE20K->20-class index for fine-tuned v1 bundles (those carry seg_classes.yaml); None for v0."""
        if not hasattr(self,"_seg_index_cache"):
            path=self.bundle/"seg_classes.yaml"
            if path.is_file():
                from Codec.edge import seg_map
                self._seg_index_cache=seg_map.build_index(seg_map.load_class_map(path))
            else:self._seg_index_cache=None
        return self._seg_index_cache
    def _one_hot_from_logits(self,logits):
        import torch
        from torch.nn import functional as F
        index=self._seg_index()
        if index is not None: # v1: the teacher emits 150 ADE20K channels; merge to the 20 codec classes
            from Codec.edge.seg_map import project_logits
            logits=project_logits(logits,index)
        ids=F.interpolate(logits,size=(256,512),mode="bilinear",align_corners=False).argmax(1)
        return F.one_hot(ids,num_classes=20).permute(0,3,1,2).float()
    def decode(self,payload):
        import torch
        if not isinstance(payload,bytes):raise TypeError("complete immutable payload bytes required")
        self.packet.decode_packet(payload) # CRC and dimensions before neural decode.
        with torch.inference_mode():result=self.model.decode(payload)
        tensor=result["reconstruction"]
        if tensor.shape!=(1,3,256,512) or not torch.isfinite(tensor).all():
            raise ValueError("invalid SQ-GAN reconstruction")
        return tensor
    def execute(self,service,job,finish_time=None):
        """Decode exactly the service-owned job, then commit success or failure."""
        if service.in_flight is not job:raise ValueError("job must be the service's in-flight dispatch")
        from time import perf_counter
        started=perf_counter()
        def finished():
            return job.started_at+perf_counter()-started if finish_time is None else finish_time
        try:
            tensor=self.decode(job.message.payload_ref)
        except Exception:
            service.complete(job.message.message_id,"failed",finished())
            raise
        service.complete(job.message.message_id,"succeeded",finished())
        return tensor


def build_scheduler_codec(config=None):
    config=config or json.loads((Path(__file__).resolve().parents[1]/"config/scheduler_codec.json").read_text())
    if config["codec"]!="sqgan":raise ValueError("scheduler codec must be the selected SQ-GAN export")
    source_root=Path(config.get("source_root",DEFAULT_ROOT))
    bundle=config.get("bundle") or source_root/config.get("bundle_relative_to_source_root",DEFAULT_BUNDLE)
    return SQGANDigitalCodec(bundle=bundle,source_root=source_root,device=config.get("device","cpu"))
