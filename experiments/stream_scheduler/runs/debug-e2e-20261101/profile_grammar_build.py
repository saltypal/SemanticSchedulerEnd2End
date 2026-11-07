import json,time
from transformers import AutoTokenizer
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_token_enforcer_tokenizer_data,build_transformers_prefix_allowed_tokens_fn)
from experiments.stream_scheduler.teacher_output import answer_schema
tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",revision="cc594898137f460bfe9f0759e9844b3ce807cfb5",local_files_only=True)
results={"old_builder_ms":[],"shared_tokenizer_builder_ms":[]}
for i in range(2):
    t=time.perf_counter();build_transformers_prefix_allowed_tokens_fn(tokenizer,JsonSchemaParser(answer_schema("prefix")))
    results["old_builder_ms"].append((time.perf_counter()-t)*1000)
t=time.perf_counter();data=build_token_enforcer_tokenizer_data(tokenizer)
results["one_time_tokenizer_data_ms"]=(time.perf_counter()-t)*1000
for i in range(2):
    t=time.perf_counter();build_transformers_prefix_allowed_tokens_fn(data,JsonSchemaParser(answer_schema("prefix")))
    results["shared_tokenizer_builder_ms"].append((time.perf_counter()-t)*1000)
results["scope"]="CPU tokenizer/grammar construction only; not GPU token generation or end-to-end teacher latency"
print(json.dumps(results),flush=True)
