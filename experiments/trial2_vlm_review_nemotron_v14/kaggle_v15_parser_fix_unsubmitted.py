import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import csv
import gc
import hashlib
import re
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from urllib.request import urlopen

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.6", "accelerate>=1.2,<2", "bitsandbytes==0.49.2", "timm", "einops", "open-clip-torch"], check=True)
import torch
import bitsandbytes as bnb
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, AutoTokenizer, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

assert torch.cuda.device_count() == 1 and "T4" in torch.cuda.get_device_name(0)
image_rows = {'42f99acd1c3529c3519264b5dd8cc857ee76000635eee85b55510f7316161643': {'publisher_url': 'https://fema-cap-imagery.s3.amazonaws.com/Images/CAP_-_KY_Flooding_July_2022/Source/22-1-5611/A0027/1_000_09_067.jpg', 'event': 'CAP_-_KY_Flooding_July_2022', 'mission': '22-1-5611', 'flight': '22-1-5611/A0027', 'split': 'calibration', 'sampling_impact_tag': False, 'image_id': '42f99acd1c3529c3519264b5dd8cc857ee76000635eee85b55510f7316161643', 'image_path': '42f99acd1c3529c3519264b5dd8cc857ee76000635eee85b55510f7316161643.jpg', 'spi_tier': 2}, 'b117edaa598114814647b06b411e5c1344e55f4822de2f257298a7fd54a7383b': {'publisher_url': 'https://fema-cap-imagery.s3.amazonaws.com/Images/CAP_-_KY_Flooding_July_2022/Source/22-1-5611/A0031/0_000_01_029.jpg', 'event': 'CAP_-_KY_Flooding_July_2022', 'mission': '22-1-5611', 'flight': '22-1-5611/A0031', 'split': 'calibration', 'sampling_impact_tag': False, 'image_id': 'b117edaa598114814647b06b411e5c1344e55f4822de2f257298a7fd54a7383b', 'image_path': 'b117edaa598114814647b06b411e5c1344e55f4822de2f257298a7fd54a7383b.jpg', 'spi_tier': 3}, 'f04dcad8ecd635d8b07752beb7548c00140d7cf91562265ad91172a8e17adda6': {'publisher_url': 'https://fema-cap-imagery.s3.amazonaws.com/Images/CAP_-_KY_Flooding_July_2022/Source/22-1-5611/A0029/0_000_05_022.jpg', 'event': 'CAP_-_KY_Flooding_July_2022', 'mission': '22-1-5611', 'flight': '22-1-5611/A0029', 'split': 'calibration', 'sampling_impact_tag': False, 'image_id': 'f04dcad8ecd635d8b07752beb7548c00140d7cf91562265ad91172a8e17adda6', 'image_path': 'f04dcad8ecd635d8b07752beb7548c00140d7cf91562265ad91172a8e17adda6.jpg', 'spi_tier': 3}, '850799bcf975e27b12eea80850689460f9ac48c73eb2c0e4c88335ab1d668cab': {'publisher_url': 'https://fema-cap-imagery.s3.amazonaws.com/Images/CAP_-_KY_Flooding_July_2022/Source/22-1-5611/A0030/0_000_02_072.jpg', 'event': 'CAP_-_KY_Flooding_July_2022', 'mission': '22-1-5611', 'flight': '22-1-5611/A0030', 'split': 'calibration', 'sampling_impact_tag': False, 'image_id': '850799bcf975e27b12eea80850689460f9ac48c73eb2c0e4c88335ab1d668cab', 'image_path': '850799bcf975e27b12eea80850689460f9ac48c73eb2c0e4c88335ab1d668cab.jpg', 'spi_tier': 2}, 'c21712966b7608b8c87b4c4efdeb1e4a5551a6a5458ec44f48fc70701b87d3a9': {'publisher_url': 'https://fema-cap-imagery.s3.amazonaws.com/Images/CAP_-_KY_Flooding_July_2022/Source/22-1-5611/A0028/1_000_06_014.jpg', 'event': 'CAP_-_KY_Flooding_July_2022', 'mission': '22-1-5611', 'flight': '22-1-5611/A0028', 'split': 'calibration', 'sampling_impact_tag': False, 'image_id': 'c21712966b7608b8c87b4c4efdeb1e4a5551a6a5458ec44f48fc70701b87d3a9', 'image_path': 'c21712966b7608b8c87b4c4efdeb1e4a5551a6a5458ec44f48fc70701b87d3a9.jpg', 'spi_tier': 3}, 'f0c4bdeef5e46fbe71558cdde2610ffd99ce99dbdd74a5c1965ab7f4c91a60e4': {'publisher_url': 'https://fema-cap-imagery.s3.amazonaws.com/Images/CAP_-_KY_Flooding_July_2022/Source/22-C-5605/1/DSC_0255.JPG', 'event': 'CAP_-_KY_Flooding_July_2022', 'mission': '22-C-5605', 'flight': '22-C-5605/1', 'split': 'calibration', 'sampling_impact_tag': True, 'image_id': 'f0c4bdeef5e46fbe71558cdde2610ffd99ce99dbdd74a5c1965ab7f4c91a60e4', 'image_path': 'f0c4bdeef5e46fbe71558cdde2610ffd99ce99dbdd74a5c1965ab7f4c91a60e4.jpg', 'spi_tier': 1}, 'c05293ad4e6b39fda0b218843ee52918cb9e005d715ec968cbe3bdbb2e8b2c28': {'publisher_url': 'https://fema-cap-imagery.s3.amazonaws.com/Images/CAP_-_KY_Flooding_July_2022/Source/22-1-5611/A0027/0_000_03_016.jpg', 'event': 'CAP_-_KY_Flooding_July_2022', 'mission': '22-1-5611', 'flight': '22-1-5611/A0027', 'split': 'calibration', 'sampling_impact_tag': True, 'image_id': 'c05293ad4e6b39fda0b218843ee52918cb9e005d715ec968cbe3bdbb2e8b2c28', 'image_path': 'c05293ad4e6b39fda0b218843ee52918cb9e005d715ec968cbe3bdbb2e8b2c28.jpg', 'spi_tier': 3}, '33ec5fab49609be6217b10f01e0a2d3b70f6efe49c38c616f7e905ee59e229f0': {'publisher_url': 'https://fema-cap-imagery.s3.amazonaws.com/Images/CAP_-_KY_Flooding_July_2022/Source/22-1-5611/A0027/1_000_02_015.jpg', 'event': 'CAP_-_KY_Flooding_July_2022', 'mission': '22-1-5611', 'flight': '22-1-5611/A0027', 'split': 'calibration', 'sampling_impact_tag': True, 'image_id': '33ec5fab49609be6217b10f01e0a2d3b70f6efe49c38c616f7e905ee59e229f0', 'image_path': '33ec5fab49609be6217b10f01e0a2d3b70f6efe49c38c616f7e905ee59e229f0.jpg', 'spi_tier': 3}, '6f66a840f6e56530c2bb82197c3a257424dd35e6add4858620002325e0495dc6': {'publisher_url': 'https://fema-cap-imagery.s3.amazonaws.com/Images/CAP_-_KY_Flooding_July_2022/Source/22-1-5611/A0029/1_000_05_072.jpg', 'event': 'CAP_-_KY_Flooding_July_2022', 'mission': '22-1-5611', 'flight': '22-1-5611/A0029', 'split': 'calibration', 'sampling_impact_tag': True, 'image_id': '6f66a840f6e56530c2bb82197c3a257424dd35e6add4858620002325e0495dc6', 'image_path': '6f66a840f6e56530c2bb82197c3a257424dd35e6add4858620002325e0495dc6.jpg', 'spi_tier': 2}, 'edc6164e5a6c0263cdfaf86d305e15236f0a119ff5918c051a94e6ad35db3fe1': {'publisher_url': 'https://fema-cap-imagery.s3.amazonaws.com/Images/CAP_-_KY_Flooding_July_2022/Source/22-1-5611/A0029/1_000_01_024.jpg', 'event': 'CAP_-_KY_Flooding_July_2022', 'mission': '22-1-5611', 'flight': '22-1-5611/A0029', 'split': 'calibration', 'sampling_impact_tag': True, 'image_id': 'edc6164e5a6c0263cdfaf86d305e15236f0a119ff5918c051a94e6ad35db3fe1', 'image_path': 'edc6164e5a6c0263cdfaf86d305e15236f0a119ff5918c051a94e6ad35db3fe1.jpg', 'spi_tier': 2}, 'a442e3245e54b4cca896a47a2d5a37b5a374ef370edc3c0bb40d0714f090ab56': {'publisher_url': 'https://fema-cap-imagery.s3.amazonaws.com/Images/CAP_-_MT_Floods_2022/Source/22-M-0210A/A0018/DSC_3153936dd535-2e01-4629-8afc-87cffcc8d8af.jpg', 'event': 'CAP_-_MT_Floods_2022', 'mission': '22-M-0210A', 'flight': '22-M-0210A/A0018', 'split': 'calibration', 'sampling_impact_tag': False, 'image_id': 'a442e3245e54b4cca896a47a2d5a37b5a374ef370edc3c0bb40d0714f090ab56', 'image_path': 'a442e3245e54b4cca896a47a2d5a37b5a374ef370edc3c0bb40d0714f090ab56.jpg', 'spi_tier': 1}, '6df24d7a0da49e9cf764c5aa0e3cc595f632163a986660b06334172563489a71': {'publisher_url': 'https://fema-cap-imagery.s3.amazonaws.com/Images/CAP_-_MT_Floods_2022/Source/22-M-0210A/A0019/DSC_03193129f985-df0e-4ffe-ac72-debdff13aed5.jpg', 'event': 'CAP_-_MT_Floods_2022', 'mission': '22-M-0210A', 'flight': '22-M-0210A/A0019', 'split': 'calibration', 'sampling_impact_tag': False, 'image_id': '6df24d7a0da49e9cf764c5aa0e3cc595f632163a986660b06334172563489a71', 'image_path': '6df24d7a0da49e9cf764c5aa0e3cc595f632163a986660b06334172563489a71.jpg', 'spi_tier': 2}, '446d99e4361cb3224dd7832e0f9427aab0ba5d2bf7e8bb2ab3800f5063fe58bc': {'publisher_url': 'https://fema-cap-imagery.s3.amazonaws.com/Images/CAP_-_MT_Floods_2022/Source/22-M-0210A/A0017/DSC_1768011f6028-077a-40d3-869c-b01dc83e6874.jpg', 'event': 'CAP_-_MT_Floods_2022', 'mission': '22-M-0210A', 'flight': '22-M-0210A/A0017', 'split': 'calibration', 'sampling_impact_tag': False, 'image_id': '446d99e4361cb3224dd7832e0f9427aab0ba5d2bf7e8bb2ab3800f5063fe58bc', 'image_path': '446d99e4361cb3224dd7832e0f9427aab0ba5d2bf7e8bb2ab3800f5063fe58bc.jpg', 'spi_tier': 1}, '46283d370824bcc089ce38f0f77de8714786ea63a0911867645471e799b5e4f3': {'publisher_url': 'https://fema-cap-imagery.s3.amazonaws.com/Images/CAP_-_MT_Floods_2022/Source/22-M-0210A/A0019/DSC_030090fdc4b7-7423-49c5-8604-c10614e09086.jpg', 'event': 'CAP_-_MT_Floods_2022', 'mission': '22-M-0210A', 'flight': '22-M-0210A/A0019', 'split': 'calibration', 'sampling_impact_tag': False, 'image_id': '46283d370824bcc089ce38f0f77de8714786ea63a0911867645471e799b5e4f3', 'image_path': '46283d370824bcc089ce38f0f77de8714786ea63a0911867645471e799b5e4f3.jpg', 'spi_tier': 2}, '1fd005af391bfe58c0858a207619c192b2751f92c461a2d9a4346c8773e6feae': {'publisher_url': 'https://fema-cap-imagery.s3.amazonaws.com/Images/CAP_-_MT_Floods_2022/Source/22-M-0210A/A0019/DSC_0251a983ee85-57dd-45c4-adc2-485922e76049.jpg', 'event': 'CAP_-_MT_Floods_2022', 'mission': '22-M-0210A', 'flight': '22-M-0210A/A0019', 'split': 'calibration', 'sampling_impact_tag': False, 'image_id': '1fd005af391bfe58c0858a207619c192b2751f92c461a2d9a4346c8773e6feae', 'image_path': '1fd005af391bfe58c0858a207619c192b2751f92c461a2d9a4346c8773e6feae.jpg', 'spi_tier': 3}}
pairs = [{'split': 'calibration', 'spi_tier': 1, 'a': '446d99e4361cb3224dd7832e0f9427aab0ba5d2bf7e8bb2ab3800f5063fe58bc', 'b': 'a442e3245e54b4cca896a47a2d5a37b5a374ef370edc3c0bb40d0714f090ab56', 'sampling_tags_differ': False}, {'split': 'calibration', 'spi_tier': 1, 'a': '446d99e4361cb3224dd7832e0f9427aab0ba5d2bf7e8bb2ab3800f5063fe58bc', 'b': 'f0c4bdeef5e46fbe71558cdde2610ffd99ce99dbdd74a5c1965ab7f4c91a60e4', 'sampling_tags_differ': True}, {'split': 'calibration', 'spi_tier': 1, 'a': 'a442e3245e54b4cca896a47a2d5a37b5a374ef370edc3c0bb40d0714f090ab56', 'b': 'f0c4bdeef5e46fbe71558cdde2610ffd99ce99dbdd74a5c1965ab7f4c91a60e4', 'sampling_tags_differ': True}, {'split': 'calibration', 'spi_tier': 2, 'a': '42f99acd1c3529c3519264b5dd8cc857ee76000635eee85b55510f7316161643', 'b': '46283d370824bcc089ce38f0f77de8714786ea63a0911867645471e799b5e4f3', 'sampling_tags_differ': False}, {'split': 'calibration', 'spi_tier': 2, 'a': '42f99acd1c3529c3519264b5dd8cc857ee76000635eee85b55510f7316161643', 'b': '6df24d7a0da49e9cf764c5aa0e3cc595f632163a986660b06334172563489a71', 'sampling_tags_differ': False}, {'split': 'calibration', 'spi_tier': 2, 'a': '42f99acd1c3529c3519264b5dd8cc857ee76000635eee85b55510f7316161643', 'b': '6f66a840f6e56530c2bb82197c3a257424dd35e6add4858620002325e0495dc6', 'sampling_tags_differ': True}, {'split': 'calibration', 'spi_tier': 2, 'a': '42f99acd1c3529c3519264b5dd8cc857ee76000635eee85b55510f7316161643', 'b': '850799bcf975e27b12eea80850689460f9ac48c73eb2c0e4c88335ab1d668cab', 'sampling_tags_differ': False}, {'split': 'calibration', 'spi_tier': 2, 'a': '42f99acd1c3529c3519264b5dd8cc857ee76000635eee85b55510f7316161643', 'b': 'edc6164e5a6c0263cdfaf86d305e15236f0a119ff5918c051a94e6ad35db3fe1', 'sampling_tags_differ': True}, {'split': 'calibration', 'spi_tier': 2, 'a': '46283d370824bcc089ce38f0f77de8714786ea63a0911867645471e799b5e4f3', 'b': '6f66a840f6e56530c2bb82197c3a257424dd35e6add4858620002325e0495dc6', 'sampling_tags_differ': True}, {'split': 'calibration', 'spi_tier': 2, 'a': '46283d370824bcc089ce38f0f77de8714786ea63a0911867645471e799b5e4f3', 'b': '850799bcf975e27b12eea80850689460f9ac48c73eb2c0e4c88335ab1d668cab', 'sampling_tags_differ': False}, {'split': 'calibration', 'spi_tier': 2, 'a': '46283d370824bcc089ce38f0f77de8714786ea63a0911867645471e799b5e4f3', 'b': 'edc6164e5a6c0263cdfaf86d305e15236f0a119ff5918c051a94e6ad35db3fe1', 'sampling_tags_differ': True}, {'split': 'calibration', 'spi_tier': 2, 'a': '6df24d7a0da49e9cf764c5aa0e3cc595f632163a986660b06334172563489a71', 'b': '6f66a840f6e56530c2bb82197c3a257424dd35e6add4858620002325e0495dc6', 'sampling_tags_differ': True}, {'split': 'calibration', 'spi_tier': 2, 'a': '6df24d7a0da49e9cf764c5aa0e3cc595f632163a986660b06334172563489a71', 'b': '850799bcf975e27b12eea80850689460f9ac48c73eb2c0e4c88335ab1d668cab', 'sampling_tags_differ': False}, {'split': 'calibration', 'spi_tier': 2, 'a': '6df24d7a0da49e9cf764c5aa0e3cc595f632163a986660b06334172563489a71', 'b': 'edc6164e5a6c0263cdfaf86d305e15236f0a119ff5918c051a94e6ad35db3fe1', 'sampling_tags_differ': True}, {'split': 'calibration', 'spi_tier': 2, 'a': '6f66a840f6e56530c2bb82197c3a257424dd35e6add4858620002325e0495dc6', 'b': '850799bcf975e27b12eea80850689460f9ac48c73eb2c0e4c88335ab1d668cab', 'sampling_tags_differ': True}, {'split': 'calibration', 'spi_tier': 2, 'a': '850799bcf975e27b12eea80850689460f9ac48c73eb2c0e4c88335ab1d668cab', 'b': 'edc6164e5a6c0263cdfaf86d305e15236f0a119ff5918c051a94e6ad35db3fe1', 'sampling_tags_differ': True}, {'split': 'calibration', 'spi_tier': 3, 'a': '1fd005af391bfe58c0858a207619c192b2751f92c461a2d9a4346c8773e6feae', 'b': '33ec5fab49609be6217b10f01e0a2d3b70f6efe49c38c616f7e905ee59e229f0', 'sampling_tags_differ': True}, {'split': 'calibration', 'spi_tier': 3, 'a': '1fd005af391bfe58c0858a207619c192b2751f92c461a2d9a4346c8773e6feae', 'b': 'b117edaa598114814647b06b411e5c1344e55f4822de2f257298a7fd54a7383b', 'sampling_tags_differ': False}, {'split': 'calibration', 'spi_tier': 3, 'a': '1fd005af391bfe58c0858a207619c192b2751f92c461a2d9a4346c8773e6feae', 'b': 'c05293ad4e6b39fda0b218843ee52918cb9e005d715ec968cbe3bdbb2e8b2c28', 'sampling_tags_differ': True}, {'split': 'calibration', 'spi_tier': 3, 'a': '1fd005af391bfe58c0858a207619c192b2751f92c461a2d9a4346c8773e6feae', 'b': 'c21712966b7608b8c87b4c4efdeb1e4a5551a6a5458ec44f48fc70701b87d3a9', 'sampling_tags_differ': False}, {'split': 'calibration', 'spi_tier': 3, 'a': '1fd005af391bfe58c0858a207619c192b2751f92c461a2d9a4346c8773e6feae', 'b': 'f04dcad8ecd635d8b07752beb7548c00140d7cf91562265ad91172a8e17adda6', 'sampling_tags_differ': False}, {'split': 'calibration', 'spi_tier': 3, 'a': '33ec5fab49609be6217b10f01e0a2d3b70f6efe49c38c616f7e905ee59e229f0', 'b': 'b117edaa598114814647b06b411e5c1344e55f4822de2f257298a7fd54a7383b', 'sampling_tags_differ': True}, {'split': 'calibration', 'spi_tier': 3, 'a': '33ec5fab49609be6217b10f01e0a2d3b70f6efe49c38c616f7e905ee59e229f0', 'b': 'c21712966b7608b8c87b4c4efdeb1e4a5551a6a5458ec44f48fc70701b87d3a9', 'sampling_tags_differ': True}, {'split': 'calibration', 'spi_tier': 3, 'a': '33ec5fab49609be6217b10f01e0a2d3b70f6efe49c38c616f7e905ee59e229f0', 'b': 'f04dcad8ecd635d8b07752beb7548c00140d7cf91562265ad91172a8e17adda6', 'sampling_tags_differ': True}, {'split': 'calibration', 'spi_tier': 3, 'a': 'b117edaa598114814647b06b411e5c1344e55f4822de2f257298a7fd54a7383b', 'b': 'c05293ad4e6b39fda0b218843ee52918cb9e005d715ec968cbe3bdbb2e8b2c28', 'sampling_tags_differ': True}, {'split': 'calibration', 'spi_tier': 3, 'a': 'b117edaa598114814647b06b411e5c1344e55f4822de2f257298a7fd54a7383b', 'b': 'c21712966b7608b8c87b4c4efdeb1e4a5551a6a5458ec44f48fc70701b87d3a9', 'sampling_tags_differ': False}, {'split': 'calibration', 'spi_tier': 3, 'a': 'b117edaa598114814647b06b411e5c1344e55f4822de2f257298a7fd54a7383b', 'b': 'f04dcad8ecd635d8b07752beb7548c00140d7cf91562265ad91172a8e17adda6', 'sampling_tags_differ': False}, {'split': 'calibration', 'spi_tier': 3, 'a': 'c05293ad4e6b39fda0b218843ee52918cb9e005d715ec968cbe3bdbb2e8b2c28', 'b': 'c21712966b7608b8c87b4c4efdeb1e4a5551a6a5458ec44f48fc70701b87d3a9', 'sampling_tags_differ': True}, {'split': 'calibration', 'spi_tier': 3, 'a': 'c05293ad4e6b39fda0b218843ee52918cb9e005d715ec968cbe3bdbb2e8b2c28', 'b': 'f04dcad8ecd635d8b07752beb7548c00140d7cf91562265ad91172a8e17adda6', 'sampling_tags_differ': True}, {'split': 'calibration', 'spi_tier': 3, 'a': 'c21712966b7608b8c87b4c4efdeb1e4a5551a6a5458ec44f48fc70701b87d3a9', 'b': 'f04dcad8ecd635d8b07752beb7548c00140d7cf91562265ad91172a8e17adda6', 'sampling_tags_differ': False}]

photos = {}
for key, row in image_rows.items():
    with urlopen(row["publisher_url"], timeout=90) as response:
        raw = response.read(32*1024*1024 + 1)
    assert len(raw) <= 32*1024*1024 and hashlib.sha256(raw).hexdigest() == key
    from PIL import ImageOps
    photo = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    photo.thumbnail((768, 768))
    photos[key] = photo
model_name = "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1"
revision = "437f4e28b989cc2d9a16b6767cc930cdf48797ff"
quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_skip_modules=["vision_model", "mlp1"],
)
question = (
    'Assess only definite visible disaster impact in this single photo. '
    'Return one JSON object with grade, extent, affected, and evidence. '
    'Grade is an integer: 0 no definite visible impact, 1 limited definite impact, '
    '2 consequential definite impact. Extent is an integer: 0 unknown or none, '
    '1 one localized location, 2 multiple distinct affected locations, '
    '3 widespread definite impact across the visible area. '
    'Affected is a list of short labels for what is visibly affected, e.g. homes, '
    'vehicles, or fields, and is empty if nothing is definitely affected. '
    'Evidence is a short string describing the visual signs and their extent. '
    'Inspect the full image, including small or partially visible signs. '
    'Do not infer damage from construction, population, nearby water, or blur. '
    'If grade is zero, use extent zero and an empty affected list. '
    'If damage is definite but its extent is unclear, use extent zero.'
)
output = Path('/kaggle/working/reviewer_extent_nemotron.json')
report = dict(model=model_name, revision=revision, gpu=torch.cuda.get_device_name(0),
              prompt=question, pairs=pairs, image_token_budget='Nemotron AutoImageProcessor default', answers=[])
def save():
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')

try:
    model = AutoModel.from_pretrained(
        model_name, revision=revision, trust_remote_code=True,
        quantization_config=quant, dtype=torch.float16, device_map={'': 0},
        attn_implementation='eager'
    ).eval()
    report['nf4_layers'] = sum(isinstance(layer, bnb.nn.Linear4bit) for layer in model.modules())
    assert report['nf4_layers'] > 0
    model.vision_model.float()
    model.vision_model.radio_model.input_conditioner.dtype = torch.float32
    model.mlp1.float()
    model.vision_model.register_forward_pre_hook(lambda module, args: (args[0].float(),))
    model.mlp1.register_forward_pre_hook(lambda module, args: (args[0].float(),))
    model.mlp1.register_forward_hook(
        lambda module, args, result: result.to(model.language_model.get_input_embeddings().weight.dtype)
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    processor = AutoImageProcessor.from_pretrained(
        model_name, revision=revision, trust_remote_code=True, device='cuda'
    )
    image_ids = sorted(photos)
    for sweep, order in enumerate((image_ids, image_ids[::-1]), start=1):
        for image_id in order:
            prompt = 'Image: <image>\n' + question
            inputs = processor([photos[image_id]])
            start = time.monotonic()
            with torch.inference_mode():
                raw = model.chat(
                    tokenizer=tokenizer, question=prompt,
                    generation_config={'max_new_tokens': 192, 'do_sample': False,
                                       'eos_token_id': tokenizer.eos_token_id},
                    **inputs
                )
            match = re.search(r'\{.*\}', raw.strip(), re.S) if isinstance(raw, str) else None
            try:
                parsed = json.loads(match[0]) if match else {}
            except json.JSONDecodeError:
                parsed = {}
            grade = parsed.get('grade')
            evidence = parsed.get('evidence', parsed.get('evidence_string'))
            extent = parsed.get('extent')
            affected = parsed.get('affected')
            valid = (
                type(grade) is int and 0 <= grade <= 2
                and type(extent) is int and 0 <= extent <= 3
                and isinstance(evidence, str) and bool(evidence.strip())
                and isinstance(affected, list)
                and all(isinstance(item, str) and item.strip() for item in affected)
                and (bool(affected) if grade else not affected and extent == 0)
            )
            report['answers'].append(dict(
                sweep=sweep, image_id=image_id, grade=grade if valid else None,
                extent=extent if valid else None, affected=affected if valid else None,
                evidence=evidence, raw=raw, valid=valid,
                seconds=time.monotonic()-start
            ))
            save()
        print(f'Completed single-photo sweep {sweep}: {len(order)} images', flush=True)
    report['status'] = 'completed'
except Exception:
    report['status'] = 'failed'
    report['error'] = traceback.format_exc()
    raise
finally:
    report['peak_gpu_bytes'] = torch.cuda.max_memory_allocated()
    save()
