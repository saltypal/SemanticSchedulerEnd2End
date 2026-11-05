"""CPU-only real processor regression: downloads tokenizer/processor, no weights."""
import argparse
import time
from pathlib import Path


def run(output):
    import sys
    import transformers
    from PIL import Image
    from transformers import AutoProcessor
    from .artifacts import write_json
    from .qwen_reviewer import build_review_messages
    from .qwen_reviewer import load_review_processor,pixel_input_receipt,thinking_template_receipt
    from .model_settings import ReviewerSettings
    from .annotation_review import REVIEW_SYSTEM_PROMPT
    started=time.perf_counter()
    revision='ebb281ec70b05090aa6165b016eac8ec08e71b17'
    print('CPU processor regression; no model weights or GPU inference',flush=True)
    from huggingface_hub import snapshot_download
    local_assets=snapshot_download('Qwen/Qwen3-VL-4B-Instruct',revision=revision,
        allow_patterns=['*.json','*.txt','*.model','*.tiktoken'])
    processor=load_review_processor(local_assets,ReviewerSettings())
    image=Image.new('RGB',(128,96),(40,120,200))
    content=[{'type':'image','image':image},{'type':'text','text':'Return one JSON object describing whether the candidate contains its required field.'}]
    legacy=[{'role':'system','content':REVIEW_SYSTEM_PROMPT},{'role':'user','content':content}]
    try:
        processor.apply_chat_template(legacy,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
    except TypeError as error:
        legacy_error={'type':type(error).__name__,'message':str(error)}
    else:
        raise AssertionError('Expected the archived string-content regression to reproduce')
    actual=processor.apply_chat_template(build_review_messages(content),tokenize=True,
                   add_generation_prompt=True,return_dict=True,return_tensors='pt')
    text_only=processor.apply_chat_template(build_review_messages(
        [{'type':'text','text':'Text-only diagnostic: compare candidate and reference JSON.'}],
        'candidate_provenance_explicit_v4'),tokenize=True,
        add_generation_prompt=True,return_dict=True,return_tensors='pt')
    assert text_only['input_ids'].numel()>0 and 'pixel_values' not in text_only
    assert actual['pixel_values'].numel()>0 and actual['input_ids'].numel()>0
    assert all(str(value.device)=='cpu' for value in actual.values() if hasattr(value,'device'))
    result={'status':'passed','scope':'actual tokenizer/image-processor only; no model inference',
        'model_revision':revision,'python':sys.executable,'transformers':transformers.__version__,
        'legacy_error_reproduced':legacy_error,
        'corrected_input_shapes':{key:list(value.shape) for key,value in actual.items() if hasattr(value,'shape')},
        'text_only_input_shapes':{key:list(value.shape) for key,value in text_only.items() if hasattr(value,'shape')},
        'effective_processor_size':dict(processor.image_processor.size),
        'pixel_receipt':pixel_input_receipt(actual,processor,512*512),
        'thinking_mode':thinking_template_receipt(processor,build_review_messages(content)),
        'seconds':time.perf_counter()-started}
    write_json(Path(output),result)
    print(result,flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    run(parser.parse_args().output)
