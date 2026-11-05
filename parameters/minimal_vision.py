"""Shared, bounded image representation for SPI, relevance and appearance novelty."""
from PIL import Image, ImageOps
import numpy as np
from .vision import FrozenVisionProvider
from .contracts import MissionSpec
from .pilot_parameters import context_similarity_key

INPUT_POLICY="RGB_longest512_bicubic_then_aspect_preserving_pad_v1"


def prepare_image(image):
    return ImageOps.contain(ImageOps.exif_transpose(image).convert("RGB"),(512,512),Image.Resampling.BICUBIC)


class MinimalVisionProvider:
    def __init__(self,checkpoint,missions,device="cpu"):
        self.provider=FrozenVisionProvider("clip",checkpoint,device=device,local_files_only=True)
        self.context_cache={}

    def context_embedding(self,mission):
        if mission.version in self.context_cache:return self.context_cache[mission.version]
        self.provider.load();p=self.provider;tokenizer=p.processor.tokenizer
        ids=tokenizer(mission.paragraph,add_special_tokens=False)["input_ids"]
        if not ids:raise ValueError("empty mission text")
        maximum=p.model.config.text_config.max_position_embeddings
        width=maximum-2;chunks=[ids[i:i+width] for i in range(0,len(ids),width)]
        if len(chunks)>16:raise ValueError("mission text exceeds the declared 16-chunk extraction budget")
        rows=[[tokenizer.bos_token_id]+c+[tokenizer.eos_token_id]+[tokenizer.pad_token_id]*(width-len(c)) for c in chunks]
        masks=[[1]*(len(c)+2)+[0]*(width-len(c)) for c in chunks]
        with p.torch.inference_mode():
            vectors=p.model.get_text_features(input_ids=p.torch.tensor(rows,device=p.device),
                attention_mask=p.torch.tensor(masks,device=p.device))
            if not p.torch.is_tensor(vectors):vectors=vectors.pooler_output
            vectors=p.torch.nn.functional.normalize(vectors.float(),dim=-1).cpu().numpy()
        if vectors.ndim!=2 or len(vectors)!=len(chunks) or not np.isfinite(vectors).all():
            raise ValueError('mission text embedding is nonfinite or has invalid shape')
        mean=np.average(vectors,axis=0,weights=[len(c) for c in chunks])
        norm=np.linalg.norm(mean)
        if not np.isfinite(norm) or norm<=1e-10:raise ValueError('mission text embedding is zero or nonfinite')
        mean=mean/norm;self.context_cache[mission.version]=mean
        return mean

    def extract_batch(self,images,missions):
        if len(images)!=len(missions):raise ValueError("image/mission batch size mismatch")
        feature_task=MissionSpec("visual_only","Extract visual features.",(("visible","a photograph"),))
        result=self.provider.extract_batch([prepare_image(i) for i in images],[feature_task]*len(images))
        for row,mission in zip(result,missions):
            row.provenance={**row.provenance,"preprocessing":INPUT_POLICY}
            try:
                context=self.context_embedding(mission)
                row.similarities={context_similarity_key(mission):float(np.asarray(row.embedding)@context)}
                row.provenance["context_method"]="token_chunk_75_weighted_normalized_mean_v1"
                row.provenance["context_status"]="ok"
            except (ValueError,FloatingPointError) as error:
                # A long/invalid paragraph must not erase independent image,
                # SPI, novelty or detector observations.
                row.similarities={}
                row.provenance.update(context_status='unavailable',context_reason=str(error))
        return result

    def close(self):self.provider.close()
