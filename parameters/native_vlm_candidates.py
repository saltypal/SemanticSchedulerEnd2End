"""Exact requested native-HF comparison checkpoints, not similarly named variants."""
from .internvl_spi_reviewer import InternVLSPIReviewer


class InternVL3Reviewer(InternVLSPIReviewer):
    model_id='OpenGVLab/InternVL3-8B-hf'
    revision='259a3b64a14623c0ec91a045cb43f7c5af5fa6af'


class Qwen25Reviewer(InternVLSPIReviewer):
    model_id='Qwen/Qwen2.5-VL-7B-Instruct'
    revision='cc594898137f460bfe9f0759e9844b3ce807cfb5'
    model_class='Qwen2_5_VLForConditionalGeneration'
    processor_kwargs={'min_pixels':65536,'max_pixels':262144}


def load_candidate(spec):
    classes={'qwen25_7b':Qwen25Reviewer,'internvl3_8b':InternVL3Reviewer}
    if spec.get('name') not in classes:raise ValueError('unsupported native comparison candidate')
    reviewer=classes[spec['name']]()
    try:reviewer.load()
    except Exception:
        reviewer.close();raise
    return reviewer
