"""Pinned native-HF InternVL candidate; shared request persistence, native loader."""
import time
from parameters.gemma_spi_reviewer import GemmaSPIReviewer

MODEL_ID='OpenGVLab/InternVL3_5-8B-HF'
REVISION='741a7d03020411e666c6109218ab71e08151ef86'


class InternVLSPIReviewer(GemmaSPIReviewer):
    thinking_template_enforced=False
    model_id=MODEL_ID
    revision=REVISION
    model_class='InternVLForConditionalGeneration'
    processor_kwargs={}

    def validate_template(self, rendered):
        if not rendered.endswith('<|im_start|>assistant\n') or '<think>' in rendered:
            raise RuntimeError('unexpected InternVL template; no R1 thinking prompt allowed')

    def load(self):
        if self.model is not None:return
        import torch, transformers
        from transformers import AutoProcessor,BitsAndBytesConfig
        from huggingface_hub import snapshot_download
        if transformers.__version__ not in ('5.3.0','5.5.0'):raise RuntimeError('native comparison requires tested Transformers5.3/5.5 runtime')
        if torch.cuda.device_count()!=2:raise RuntimeError('two GPUs required; GPU1 owns reviewer')
        path=snapshot_download(self.model_id,revision=self.revision,
            allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'])
        self.torch=torch;started=time.perf_counter()
        self.processor=AutoProcessor.from_pretrained(path,local_files_only=True,**self.processor_kwargs)
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
        self.model=getattr(transformers,self.model_class).from_pretrained(path,local_files_only=True,
            dtype=torch.float16,device_map={'':1},quantization_config=quant,attn_implementation='sdpa').eval()
        devices=sorted({str(p.device) for p in self.model.parameters()})
        if devices!=['cuda:1']:raise RuntimeError('InternVL device ownership failed: '+str(devices))
        self.finite_checked_modes=set()
        self.provenance={'model_id':self.model_id,'revision':self.revision,'transformers':transformers.__version__,
            'parameter_devices':devices,'quantization':'NF4_double_quant_FP16',
            'load_seconds':time.perf_counter()-started,'model_footprint_bytes':self.model.get_memory_footprint(),
            'thinking_control':'no_R1_prompt_requested; generated_thought_markers_rejected',
            'template_enable_thinking_supported':False,'mode':'image_only_SPI_pseudo_labels',
            'image_processor':self.processor.image_processor.to_dict(),
            'independent_ground_truth':False}


def load_internvl(spec):
    reviewer=InternVLSPIReviewer()
    try:reviewer.load()
    except Exception:
        reviewer.close();raise
    return reviewer
