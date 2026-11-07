# Using the implemented scheduler

For an executable terminal walkthrough, run `python scripts/demo_decoder_scheduler.py`.
See [TERMINAL_DEMO.md](TERMINAL_DEMO.md) for cached versus live extraction modes,
saved JSON traces and the Qwen annotation-usefulness correction.

The runtime ranks complete digital messages before decoding. The experiment did
not encode or reconstruct images. A real codec is still supplied by the teammate;
no tensor slice or continuous DeepJSCC/IQ stream is treated as an independently
decodable digital packet.

## Load an evaluated model

Run from the repository root. The example loads the model selected on validation
C, not a model chosen by looking at the test scores:

```python
from Scheduler.decoder_scheduler import load_scheduler
from SemParam.decoder_parameters import PROFILES
from INFRA.decoder_queue import DecoderQueue

artifact = (
    "results/disaster_decoder_v3/20261023_stageB_v1/full/study/"
    "building_damage/selected_scheduler"
)
scheduler = load_scheduler(artifact, PROFILES)
receiver = DecoderQueue(scheduler, 1001)
```

The corresponding directories for roads and flooding contain their own
validation-selected models. One active task/profile is supported per queue.
Changing a producer model, task rubric or calibration requires a compatible
profile and renewed validation; matching a numeric profile ID alone does not
authenticate a sender or attest its model version.

## Extract the header when an observation arrives

This function illustrates the existing APIs. `complete_payload_ref` is a real,
already-encoded complete digital bundle supplied by the codec/transport layer;
this example does not create a fake latent payload. The caller supplies an open
PIL image, source/session identity and a unique uint32 message token.

```python
import json
from pathlib import Path
from SemParam.decoder_extractors import FrozenImageExtractor
from SemParam.decoder_parameters import (
    ParameterExtractor, ParameterHeader, TemperatureCalibration,
    TransmitterMemory, get_profile,
)
from INFRA.decoder_queue import ReceivedMessage

profile = get_profile(1001)
image_models = FrozenImageExtractor(device="auto")
parameter_extractor = ParameterExtractor()
calibration = TemperatureCalibration(**json.loads(Path(
    "results/disaster_decoder_v3/20261023_stageB_v1/full/study/"
    "building_damage/calibration.json"
).read_text()))
memories = {}

def admit_image(image, complete_payload_ref, source_id, session_id,
                token, arrival_time, sequence):
    key = (source_id, session_id, profile.profile_id)
    memory = memories.setdefault(key, TransmitterMemory(source_id, profile.profile_id))
    evidence = image_models.extract(image, f"{source_id}:{session_id}:{token}")
    # No trusted capture timestamp is invented: F is unavailable here.
    parameters = parameter_extractor.extract(evidence, profile, {}, memory, calibration)
    header = ParameterHeader(profile.profile_id, token, memory.version, 0,
                             parameters.values, parameters.available)
    wire = header.to_bytes()
    memory.commit(evidence)  # only after querying history and producing the header
    receiver.enqueue(ReceivedMessage(
        source_id, session_id, token, wire, complete_payload_ref,
        True, arrival_time, sequence,
    ))
```

This image-to-header path is runtime extraction. It does not depend on a dataset
cache. Dataset caches were used only to avoid rerunning frozen image models for
every training epoch or policy comparison. Scarcity and novelty were regenerated
causally for each experimental UE history, not stored as one universal header
per image. Qwen is an offline annotation assistant and is not invoked here.

## Dispatch and decode

```python
selected = receiver.dispatch(now=queue_clock)
if selected is not None:
    decision, message = selected
    reconstruction = teammate_codec.decode(message.payload_ref)
```

`queue_clock` and arrival times use the same nondecreasing clock. A trusted
`utc_now_ms` may additionally update an available F field. Dispatch never changes
R/S/Q/I/P/C/N. Only INFRA owns enqueue/removal; the policy receives header values,
availability bits and legal identity/tie information, never the payload reference.

The codec call above is the integration handoff, not an implemented or measured
reconstruction result. The codec must establish checkpoint/shape compatibility,
all hyperlatent/side-information dependencies, exact payload bytes and service
time before a real decoded-evidence benchmark is reported. Transport must establish
completeness and source authenticity. CRC32 does not provide authentication.

## Reproduce the study

The private Kaggle notebooks are `hmmmkk/disaster-eight-field-ollama` (frozen
extraction), `hmmmkk/disaster-eight-field-scheduler` (original full training), and
`hmmmkk/disaster-eight-field-final-notebook` (resumed verification and executed
notebook export). The final notebook metadata attaches the first two kernel
outputs. Building it does not submit it:

```powershell
python scripts/build_decoder_v3_notebook.py --stage final
```

The final notebook reused the verified extraction cache and completed training
checkpoints. It required all 39 quality-metric rows to match the original run
within 1e-12, repeated exported-model prediction checks, and saved actual captured
execution outputs. It did not choose a new method or seed from test performance.
See the artifact index for the verified executed file and exact run versions.
